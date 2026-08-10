import logging
from typing import List, Dict, Tuple
from app.services.vector_db import vector_db_service
from app.services.docscan import BGE_M3_BASELINE, adjust_similarity
from app.utils.document_parser import DocumentParser
from app.core.config import settings

logger = logging.getLogger(__name__)


class ScannerService:
    def __init__(self):
        self.parser = DocumentParser()
        self._initialized = False

    def _ensure_initialized(self):
        """延迟初始化 vector_db_service,避免 import 时 sys.path 尚未配置."""
        if not self._initialized:
            vector_db_service.initialize()
            self._initialized = True

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def scan_text(self, text: str) -> Dict:
        self._ensure_initialized()
        chunks = self.parser._split_into_chunks(text)
        if not chunks:
            return self._empty_result()

        # 构造 chunk list
        chunk_dicts = [
            {"content": chunk, "source": "", "page": i + 1, "doc_type": "text"}
            for i, chunk in enumerate(chunks)
        ]
        return self._scan_chunks(chunk_dicts)

    def _scan_chunks(self, chunks: List[Dict]) -> Dict:
        """批量扫描核心 — 批量嵌入 + 批量 ChromaDB 查询."""

        from app.services.ingestion import data_ingestion_service, create_embedder_from_settings

        # 1. 批量嵌入所有 chunk 文本 (一次性, 而非逐个)
        chunk_texts = [chunk["content"] for chunk in chunks]

        # 使用 Indexer 的 embedder (与入库一致); 取不到时 fallback 创建临时 embedder。
        # fallback embedder 用完必须 release(), 否则 MPSEmbedder 模型权重(~2GB) 驻留
        # GPU 直到进程退出 (OllamaEmbedder 的 SQLite 缓存连接也会泄漏)。
        _fallback_embedder = None
        try:
            indexer = data_ingestion_service._get_indexer()
            embedder = indexer.embedder
        except Exception:
            # fallback: 与 Indexer._create_embedder 判序一致的工厂函数 (含 ollama_cache_path)
            _fallback_embedder = create_embedder_from_settings()
            embedder = _fallback_embedder

        # ★ 批量嵌入: 一次调用 embed() 传入所有文本, OllamaEmbedder 内部会用
        # /api/embed 批量接口, 100个一批; MPSEmbedder 16个一批; 避免逐个嵌入
        try:
            all_embeddings = embedder.embed(chunk_texts)
        finally:
            if _fallback_embedder is not None:
                try:
                    _fallback_embedder.release()
                except Exception:
                    pass
                _fallback_embedder = None

        # 2. 批量 ChromaDB 查询 — 一次传全部 query_embeddings
        # ChromaDB collection.query() 支持多个 query_embeddings,
        # 返回每个 query 的 top-n 结果, 比逐个 query 快得多
        import math
        for level in ("confidential", "restricted"):
            collection = vector_db_service._get_collection(level)
            if not collection:
                continue
            count = collection.count()
            if count == 0:
                continue
            # 与 DocScan (n_results=10) 对齐: 原 min(3, count) 召回过低, 敏感 chunk
            # 若排在 top4-10 会被整体漏检 (false negative)。
            actual_n = min(10, count)
            # ★ 批量查询: 所有 chunk 的 embeddings 一次传入
            batch_results = collection.query(
                query_embeddings=all_embeddings,
                n_results=actual_n,
                include=["documents", "metadatas", "distances"],
            )
            # batch_results 结构:
            #   documents:  [[query1_top1, query1_top2, ...], [query2_top1, ...], ...]
            #   distances:  [[d1_1, d1_2, ...], [d2_1, ...], ...]
            #   metadatas:  [[m1_1, m1_2, ...], [m2_1, ...], ...]
            # 每个 query_i 对应 chunks[i] 的搜索结果

            # 存储批量结果供后续 _process_similar_results 使用
            for i, chunk in enumerate(chunks):
                chunk_text = chunk["content"]
                if not batch_results.get("documents") or i >= len(batch_results["documents"]):
                    continue
                docs = batch_results["documents"][i] or []
                dists = batch_results.get("distances", [[]])[i] or []
                metas = batch_results.get("metadatas", [[]])[i] or []

                similar_items = []
                for j, doc in enumerate(docs):
                    distance = dists[j] if j < len(dists) else 2.0
                    raw_sim = max(0.0, min(1.0, 1.0 - distance / 2.0))
                    similar_items.append({
                        "content": doc,
                        "metadata": metas[j] if j < len(metas) else {},
                        "similarity": raw_sim,
                    })

                # 按相似度降序排列
                similar_items.sort(key=lambda x: x["similarity"], reverse=True)

                # 存入 chunk dict 供后续处理
                if "_similar_results" not in chunk:
                    chunk["_similar_results"] = {}
                chunk["_similar_results"][level] = similar_items

        # 3. 处理结果 (与旧版 _process_similar_results 一致)
        all_segments: List[dict] = []
        chunk_results = []

        for chunk in chunks:
            chunk_text = chunk["content"]
            similar_results = chunk.get("_similar_results", {})
            segs, has_conf, has_rest = self._process_similar_results(chunk_text, similar_results)
            all_segments.extend(segs)

            chunk_results.append({
                "chunk_index": chunk.get("page", 0),
                "source": chunk.get("source", ""),
                "has_confidential": has_conf,
                "has_restricted": has_rest,
                "segments": segs,
                "full_content": chunk_text,
            })

        summary = self._summary_by_level(all_segments)
        return {
            "total_chunks": len(chunks),
            "has_sensitive": len(all_segments) > 0,
            "segments": all_segments,
            "chunk_results": chunk_results,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # 内部辅助 (消除 confidential/restricted 重复逻辑)
    # ------------------------------------------------------------------

    def _process_similar_results(
        self,
        chunk_text: str,
        similar_results: Dict,
    ) -> Tuple[List[dict], bool, bool]:
        """处理 confidential + restricted 搜索结果,返回 (segments, has_conf, has_rest).

        ★ 基线调整: bge-m3 不相关文本基线 ≈ 0.37, 为让相似度更直观,
        将有效量程 [0.37, 1.0] 重标定到 [0, 1], 与 DocScan 比对一致。
        阈值也应相应调整 (adjusted_threshold = adjust_similarity(original_threshold))。
        """
        segments: List[dict] = []
        has_conf = False
        has_rest = False

        for level, threshold_key in [
            ("confidential", "SIMILARITY_THRESHOLD_CONFIDENTIAL"),
            ("restricted", "SIMILARITY_THRESHOLD_RESTRICTED"),
        ]:
            # 将配置阈值也做基线调整, 与 adjusted_similarity 在同一尺度上比较
            original_threshold = getattr(settings, threshold_key)
            adjusted_threshold = adjust_similarity(original_threshold)
            for result in similar_results.get(level, []):
                raw_sim = result["similarity"]
                adjusted_sim = adjust_similarity(raw_sim)
                if adjusted_sim >= adjusted_threshold:
                    if level == "confidential":
                        has_conf = True
                    else:
                        has_rest = True
                    positions = self._find_matching_positions(chunk_text, result["content"])
                    for start, end in positions:
                        segments.append({
                            "content": chunk_text[start:end],
                            "start": start,
                            "end": end,
                            "level": level,
                            "confidence": adjusted_sim,
                            "raw_similarity": raw_sim,
                            "matched_source": result["metadata"].get("source", ""),
                            "matched_content": result["content"][:200],
                        })
        return segments, has_conf, has_rest

    def _empty_result(self) -> Dict:
        return {
            "total_chunks": 0,
            "has_sensitive": False,
            "segments": [],
            "chunk_results": [],
            "summary": {"confidential": 0, "restricted": 0},
        }

    def _summary_by_level(self, segments: List[dict]) -> Dict:
        counts = {"confidential": 0, "restricted": 0}
        for seg in segments:
            level = seg.get("level", "")
            if level in counts:
                counts[level] += 1
        return counts

    # ------------------------------------------------------------------
    # 位置匹配
    # ------------------------------------------------------------------

    def _find_matching_positions(self, text: str, pattern: str, threshold: float = 0.7) -> List[Tuple[int, int]]:
        positions = []

        if not pattern or not text:
            return positions

        # 预计算清理后的 pattern 词集合（避免每词重复清理）
        def _clean(s: str) -> str:
            return ''.join(c for c in s if c.isalnum())

        # Word-level splitting; fallback to character-level for Chinese/continuous text
        pattern_words = set(pattern.lower().split())
        if len(pattern_words) <= 1 and len(pattern) > 1:
            # No spaces in pattern — likely Chinese or continuous text; match by character
            pattern_chars = list(pattern.lower())
            cleaned_pattern = {_clean(c) for c in pattern_chars if _clean(c)}
        else:
            cleaned_pattern = {_clean(p) for p in pattern_words if _clean(p) and len(_clean(p)) > 2}
        if not cleaned_pattern:
            return positions

        words = text.lower().split()
        if len(words) <= 1 and len(text) > 1:
            # No spaces in text — likely Chinese or continuous text; treat each character as a "word"
            words = list(text.lower())

        # 预清理所有 text 词（每词只清一次）
        cleaned_words = [_clean(w) for w in words]
        matching_indices = []

        for i, word_clean in enumerate(cleaned_words):
            if not word_clean:
                continue
            for pw_clean in cleaned_pattern:
                if pw_clean in word_clean or word_clean in pw_clean:
                    matching_indices.append(i)
                    break

        # 无匹配词时不返回假段,返回空列表 (修复原 (0,200) 误报)
        if not matching_indices:
            return positions

        # 构建每个词在原始 text 中的真实字符位置
        word_positions: List[Tuple[int, int]] = []
        search_offset = 0
        lower_text = text.lower()
        for word in words:
            # 精确查找该词在当前偏移之后的位置 (避免重复词偏移错误)
            found_pos = lower_text.find(word, search_offset)
            if found_pos == -1:
                # 退路: 使用 search_offset 估算
                found_pos = search_offset
            word_positions.append((found_pos, found_pos + len(word)))
            search_offset = found_pos + len(word)

        clusters = []
        current_cluster = [matching_indices[0]]

        for idx in matching_indices[1:]:
            if idx - current_cluster[-1] <= 5:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        clusters.append(current_cluster)

        for cluster in clusters:
            if len(cluster) >= max(1, int(len(pattern_words) * threshold)):
                start_pos = word_positions[cluster[0]][0]
                end_pos = word_positions[cluster[-1]][1]
                positions.append((start_pos, min(end_pos, len(text))))

        return positions

scanner_service = ScannerService()