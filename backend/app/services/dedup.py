"""数据去重服务 - 跨密级文档相似度比对.

功能:
1. 受限 vs 公开: 找出受限文档中与公开文档相似的内容
2. 机密 vs 公开: 找出机密文档中与公开文档相似的内容
3. 机密 vs 受限: 找出机密文档中与受限文档相似的内容

输出:
- 三组比对结果统计
- 详细数据写入 Markdown 文件

关键设计:
1. 只使用子块(child chunk)做比对: 子块粒度细(~500字)，比对精准，
   避免父块与子块内容重叠导致匹配数虚高。
2. 从 ChromaDB 同时读取入库时已存储的向量和对应的文本/元数据，
   保证向量与文本严格对齐，且两次比对结果完全一致（确定性）。
3. 匹配详情中附带父块上下文，便于用户理解匹配内容的完整上下文。
4. 基线调整: bge-m3 不相关文本基线 ≈ 0.37, 将有效量程重标定到 [0,1] (线性映射)。
   DocScan 用 √ 映射 (adjust_similarity_docscan), 两者单调但阈值不可互换。
"""

from __future__ import annotations

import json
import os
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.services.docscan import (
    BGE_M3_BASELINE,
    text_ngram_overlap,
    longest_matching_run,
    build_ngram_index,
)
from app.services.vector_db import vector_db_service


class DedupResult:
    """去重比对结果."""

    def __init__(self):
        self.pair_name: str = ""
        self.high_similarity_count: int = 0
        self.medium_similarity_count: int = 0
        self.low_similarity_count: int = 0
        self.total_checked: int = 0
        self.total_comparisons_in_pair: int = 0  # source_count × target_count
        self.matches: List[Dict[str, Any]] = []
        self.elapsed_seconds: float = 0.0
        self.cancelled: bool = False
        self.source_ids: List[str] = []   # source 分类的所有 chunk IDs (与 source_data.ids 对齐)
        self.target_ids: List[str] = []   # target 分类的所有 chunk IDs


class ChunkData:
    """一个分类的全部 chunk 数据（向量 + 文本 + 元数据），保证对齐."""

    def __init__(self, embeddings: np.ndarray, texts: List[str], metadatas: List[dict],
                 ids: Optional[List[str]] = None):
        self.embeddings = embeddings  # (N, dim) 已归一化
        self.texts = texts            # 长度 N
        self.metadatas = metadatas    # 长度 N
        self.ids = ids or []          # 长度 N, ChromaDB chunk IDs
        self.count = len(texts)


class DataDeduplicationService:
    """数据去重服务."""

    def __init__(self):
        self._embedder = None
        # 写入 dedup_results 目录前会清空旧文件;并发 run_deduplication / dedup/apply
        # 会互相删掉对方刚写的产物。用锁串行化写阶段。
        import threading as _threading
        self._write_lock = _threading.Lock()

    def _get_embedder(self):
        """获取 embedder (仅作为 fallback)."""
        if self._embedder is not None:
            return self._embedder

        # MD2RAG path is configured in app/main.py at startup; no need to manipulate sys.path here.
        from app.services.ingestion import data_ingestion_service
        indexer = data_ingestion_service._get_indexer()
        self._embedder = indexer.embedder
        print(f"[DEDUP] Using embedder (fallback): {type(self._embedder).__name__}, dim={self._embedder.dimension}")
        return self._embedder

    def _count_chunks(self, classification: str) -> int:
        """获取指定密级的子块数量（不拉取全量数据）.

        只统计子块，因为去重比对只使用子块。
        ★ 复用 vector_db_service 的共享 ChromaDB client,避免多客户端锁竞争。
        """
        try:
            collection_name = f"md2rag_{classification}_child"
            collection = vector_db_service.get_collection(collection_name)
            if collection is None:
                return 0
            return collection.count()
        except Exception as e:
            print(f"[DEDUP] Error counting child chunks for {classification}: {e}")
            return 0

    def _load_chunk_data_from_chromadb(self, classification: str) -> Optional[ChunkData]:
        """从 ChromaDB 读取指定分类的子块向量、文本和元数据.

        只读取子块(_child)集合，因为:
        1. 子块粒度细(~500字)，比对更精准
        2. 避免父块和子块内容重叠导致匹配数虚高
        3. 子块可通过 parent_doc_id 关联父块上下文

        同时预加载父块文本，写入匹配详情时附带父块上下文。
        ★ 复用 vector_db_service 的共享 ChromaDB client,避免多客户端锁竞争。
        """
        try:
            client = vector_db_service.get_client()

            # --- 读取子块 ---
            child_collection_name = f"md2rag_{classification}_child"
            collection = client.get_collection(child_collection_name)
            result = collection.get(
                include=["embeddings", "documents", "metadatas"],
            )

            if not result or result.get("embeddings") is None or len(result["embeddings"]) == 0:
                print(f"[DEDUP] No child chunks found for {classification}")
                return None

            child_embeddings = np.array(result["embeddings"], dtype=np.float32)
            child_texts = result.get("documents") or [""] * len(result["ids"])
            child_metadatas = result.get("metadatas") or [{}] * len(result["ids"])

            print(f"[DEDUP] Read {len(result['ids'])} child chunks from {child_collection_name} "
                  f"(dim={child_embeddings.shape[1]})")

            # --- 预加载父块文本（用于匹配详情中的上下文） ---
            parent_texts = {}
            try:
                parent_collection_name = f"md2rag_{classification}_parent"
                parent_col = client.get_collection(parent_collection_name)
                parent_result = parent_col.get(
                    include=["documents", "metadatas"],
                )
                if parent_result and parent_result.get("documents"):
                    parent_metas = parent_result.get("metadatas") or [{}] * len(parent_result["ids"])
                    for idx, pid in enumerate(parent_result["ids"]):
                        doc_id = parent_metas[idx].get("doc_id", pid)
                        parent_texts[doc_id] = parent_result["documents"][idx] or ""
                    print(f"[DEDUP] Loaded {len(parent_texts)} parent texts for {classification}")
            except Exception as e:
                print(f"[DEDUP] Cannot load parent texts for {classification}: {e}")

            # 将父块上下文注入子块 metadata
            for meta in child_metadatas:
                parent_doc_id = meta.get("parent_doc_id", "")
                if parent_doc_id and parent_doc_id in parent_texts:
                    meta["_parent_text"] = parent_texts[parent_doc_id]

            # 归一化
            norms = np.linalg.norm(child_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            child_embeddings = child_embeddings / norms

            print(f"[DEDUP] Child chunks for {classification}: "
                  f"{child_embeddings.shape[0]} vectors, {len(child_texts)} texts")
            return ChunkData(child_embeddings, child_texts, child_metadatas, ids=result["ids"])

        except Exception as e:
            print(f"[DEDUP] Error loading chunk data for {classification}: {e}")
            return None

    def _load_chunk_data_by_embedding(self, classification: str) -> Optional[ChunkData]:
        """Fallback: 从 ChromaDB 读取子块文本，重新嵌入生成向量.

        仅在 ChromaDB 存储向量不可用时使用。
        只加载子块。
        ★ 复用 vector_db_service 的共享 ChromaDB client,避免多客户端锁竞争。
        """
        try:
            client = vector_db_service.get_client()

            child_collection_name = f"md2rag_{classification}_child"
            collection = client.get_collection(child_collection_name)
            result = collection.get(include=["documents", "metadatas"])

            if not result or not result.get("documents"):
                return None

            texts = result["documents"]
            metadatas = result.get("metadatas") or [{}] * len(texts)

            # 重新嵌入
            embedder = self._get_embedder()
            embeddings = np.array(embedder.embed(texts), dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

            print(f"[DEDUP] Re-embedded {len(texts)} child chunks for {classification}")
            return ChunkData(embeddings, texts, metadatas, ids=result["ids"])

        except Exception as e:
            print(f"[DEDUP] Error in fallback loading for {classification}: {e}")
            return None

    def _compare_collections(
        self,
        source_data: ChunkData,
        target_data: ChunkData,
        source_cls: str,
        target_cls: str,
        max_samples: int = 0,
        progress_callback: Optional[Callable] = None,
        completed_before: int = 0,
        total_comparisons: int = 0,
        pair_label: str = "",
        cancel_event=None,
    ) -> DedupResult:
        """比对两个密级的数据，找出相似内容.

        使用预加载的 ChunkData（向量+文本+元数据已对齐）。
        结果按相似度自动分为三档：≥0.8 高 / 0.65-0.8 中 / 0.5-0.65 弱
        """
        result = DedupResult()
        result.pair_name = f"{source_cls}-{target_cls}"
        start_time = time.time()

        source_count = source_data.count
        target_count = target_data.count
        pair_comparisons = source_count * target_count

        print(f"[DEDUP] {result.pair_name}: source={source_count}, target={target_count}, "
              f"comparisons={pair_comparisons}")

        if source_count == 0 or target_count == 0:
            result.total_checked = pair_comparisons
            result.total_comparisons_in_pair = pair_comparisons
            result.elapsed_seconds = time.time() - start_time
            if progress_callback:
                progress_callback({
                    "phase": "progress",
                    "message": f"完成比对: {pair_label}, 无数据需比对",
                    "progress": (completed_before + pair_comparisons) / max(total_comparisons, 1),
                    "completed": completed_before + pair_comparisons,
                    "total": max(total_comparisons, completed_before + pair_comparisons),
                })
            return result

        # 采样
        source_embeddings = source_data.embeddings
        source_texts = source_data.texts
        source_metadatas = source_data.metadatas
        source_chunk_ids = source_data.ids
        target_embeddings = target_data.embeddings
        target_texts = target_data.texts
        target_metadatas = target_data.metadatas
        target_chunk_ids = target_data.ids

        if max_samples > 0:
            if source_count > max_samples:
                indices = sorted(random.sample(range(source_count), max_samples))
                source_embeddings = source_embeddings[indices]
                source_texts = [source_texts[i] for i in indices]
                source_metadatas = [source_metadatas[i] for i in indices]
                source_chunk_ids = [source_chunk_ids[i] for i in indices] if source_chunk_ids else []
                source_count = len(source_texts)
            if target_count > max_samples:
                indices = sorted(random.sample(range(target_count), max_samples))
                target_embeddings = target_embeddings[indices]
                target_texts = [target_texts[i] for i in indices]
                target_metadatas = [target_metadatas[i] for i in indices]
                target_chunk_ids = [target_chunk_ids[i] for i in indices] if target_chunk_ids else []
                target_count = len(target_texts)
            pair_comparisons = source_count * target_count

        if total_comparisons == 0:
            total_comparisons = pair_comparisons

        # 批量比对
        BATCH_SIZE = 200
        all_matches = []
        pair_completed = 0
        # 预建每个 target 的 n-gram set, 避免在 match 循环内对同一 target 反复重建
        # (同一 target 可能被多个 source 命中, 原 build_ngram_index([target_text]) 逐对重建)
        target_ngram_sets = [build_ngram_index([t]) for t in target_texts]

        for batch_start in range(0, source_count, BATCH_SIZE):
            # 协作式取消检查
            if cancel_event and cancel_event.is_set():
                result.matches = all_matches
                result.total_checked = pair_comparisons
                result.total_comparisons_in_pair = pair_comparisons
                result.elapsed_seconds = time.time() - start_time
                result.cancelled = True
                if progress_callback:
                    progress_callback({
                        "phase": "cancelled",
                        "message": f"去重比对被用户取消: {pair_label}",
                        "progress": (completed_before + pair_completed) / total_comparisons,
                        "completed": completed_before + pair_completed,
                        "total": total_comparisons,
                    })
                return result

            batch_end = min(batch_start + BATCH_SIZE, source_count)
            batch_vecs = source_embeddings[batch_start:batch_end]

            # 计算相似度矩阵: (batch_size, dim) @ (dim, N_target)
            sim_batch = batch_vecs @ target_embeddings.T

            # 向量化重标定 + 阈值筛选, 替代原先逐 (j,k) 的 Python 双重循环
            # (大集合如 2196×2741≈600 万对, 纯 Python 循环 + 逐对调 adjust_similarity 很慢)。
            # adjust_similarity 是单调线性映射 max(0,(raw-B)/(1-B)), 故 adjusted>=0.5
            # 等价于 raw>=B+0.5*(1-B); 直接对整个矩阵做 numpy 掩码, 只对命中的少数对
            # 跑 n-gram 合议与分桶。结果与原实现逐位等价 (IEEE754 逐元素运算一致),
            # 命中对顺序与原 j 外层/k 内层循环一致 (argwhere 返回 C-order)。
            # 注意: sim_batch 是 float32 (源/目标向量均为 float32)。原实现用 float(sim)
            # 把每个值提升为 float64 再算 adjust; 为逐位等价这里也先转 float64, 否则
            # float32 运算会在 ~第8位有效数字出现差异, 可能把边界对 (adjusted≈0.5/0.65/0.8)
            # 分到不同桶或在阈值处翻转匹配集。
            adjusted_batch = np.maximum(
                0.0, (sim_batch.astype(np.float64) - BGE_M3_BASELINE) / (1.0 - BGE_M3_BASELINE)
            )
            # 跳过零向量行 (与原 np.linalg.norm(batch_vecs[j])==0 一致)
            batch_norms = np.linalg.norm(batch_vecs, axis=1)
            match_mask = (adjusted_batch >= 0.5) & (batch_norms != 0)[:, None]

            for j, k in np.argwhere(match_mask):
                source_idx = batch_start + int(j)
                similarity = float(sim_batch[j, k])
                adjusted_sim = float(adjusted_batch[j, k])
                # 结果按 adjusted 三档分类:
                # ≥0.8: 高度相似(几乎确定是重复)
                # 0.5-0.8: 中度相似(主题相近, 需要人工确认)
                # <0.5: 弱相关(语义有交叠但不算重复) -- 已被掩码过滤
                # 注: 这里 adjusted 是线性映射 (raw-0.37)/0.63; DocScan 用 √ 映射
                # (adjust_similarity_docscan), 两者单调但数值不等价, 阈值不可互换。
                source_meta = source_metadatas[source_idx]
                target_meta = target_metadatas[int(k)]
                source_text = source_texts[source_idx]
                target_text = target_texts[int(k)]

                # ★ B1: n-gram 合议 - 高向量但字面 overlap 极低的可能是
                # boilerplate / 同领域不同内容, 不该自动列入"准备去重"。
                # 这里逐对算 (query=source, target=target) 的字面信号。
                text_overlap = text_ngram_overlap(source_text, target_text)
                # 单文档内 longest_run: 用预建的 target n-gram set (避免逐对 build_ngram_index)
                target_grams = target_ngram_sets[int(k)]
                longest_run = longest_matching_run(source_text, target_grams)

                match_info = {
                    "source_chunk_id": source_chunk_ids[source_idx] if source_chunk_ids else "",
                    "source_text": source_text[:300],
                    "target_text": target_text[:300],
                    "source_parent_text": (source_meta.get("_parent_text") or "")[:300],
                    "target_parent_text": (target_meta.get("_parent_text") or "")[:300],
                    "similarity": similarity,
                    "adjusted_similarity": adjusted_sim,
                    "text_overlap": round(text_overlap, 4),
                    "longest_run": longest_run,
                    "source_metadata": source_meta,
                    "target_metadata": target_meta,
                }
                all_matches.append(match_info)

                # 三档分桶 - B1: 高桶要求字面也支持 (overlap≥0.3 或 run≥30)
                # 否则降级为中等, 避免纯向量 boilerplate 被列入"准备去重"
                ngram_supports_high = (text_overlap >= 0.3 or longest_run >= 30)
                if adjusted_sim >= 0.8 and ngram_supports_high:
                    result.high_similarity_count += 1
                elif adjusted_sim >= 0.65:
                    result.medium_similarity_count += 1
                else:
                    result.low_similarity_count += 1

            pair_completed += (batch_end - batch_start) * target_count
            total_completed = completed_before + pair_completed
            if progress_callback:
                progress_callback({
                    "phase": "progress",
                    "message": f"正在比对: {pair_label} ({batch_end}/{source_count})",
                    "progress": total_completed / total_comparisons,
                    "completed": total_completed,
                    "total": total_comparisons,
                })

        result.matches = all_matches
        result.total_checked = pair_comparisons
        result.total_comparisons_in_pair = pair_comparisons
        result.elapsed_seconds = time.time() - start_time
        result.source_ids = source_chunk_ids
        result.target_ids = target_chunk_ids
        print(f"[DEDUP] {result.pair_name}: found {len(all_matches)} matches in {result.elapsed_seconds:.1f}s")
        return result

    def run_deduplication(
        self,
        max_samples: int = 0,
        progress_callback: Optional[Callable] = None,
        cancel_event=None,
        dedup_threshold: float = 0.75,
        auto_dedup: bool = False,
    ) -> Dict[str, Any]:
        """执行完整的数据去重比对.

        结果按相似度自动分为三档：
        - ≥0.8: 高度相似(几乎确定是重复)
        - 0.65-0.8: 中度相似(主题相近, 需要人工确认)
        - 0.5-0.65: 弱相关(语义有交叠但不算重复)

        Args:
            max_samples: 最大采样数
            progress_callback: 进度回调,接收 dict (含 completed/total 字段)
            dedup_threshold: 去重相似度阈值, 超过此阈值的匹配将写入"准备去重.md"
            auto_dedup: 是否自动执行去重删除(从高密级库删除超过阈值的 chunk)
        """
        overall_start = time.time()

        # 三组比对
        comparisons = [
            ("restricted", "public", "受限-公开"),
            ("confidential", "public", "机密-公开"),
            ("confidential", "restricted", "机密-受限"),
        ]

        # 预先统计各组 chunk 数量，计算总比对次数
        # 公式: 总比对次数 = 受限×公开 + 机密×公开 + 机密×受限
        chunk_counts = {}
        total_comparisons = 0
        for source_cls, target_cls, label in comparisons:
            source_count = self._count_chunks(source_cls)
            target_count = self._count_chunks(target_cls)
            pair_total = source_count * target_count
            chunk_counts[label] = {"source": source_count, "target": target_count, "pair_total": pair_total}
            total_comparisons += pair_total

        if total_comparisons == 0:
            total_comparisons = 1

        print(f"[DEDUP] Total comparisons needed: {total_comparisons} "
              f"(details: {chunk_counts})")

        # 预加载所有分类的 chunk 数据（向量+文本+元数据，每个分类只读一次）
        unique_classifications = set()
        for source_cls, target_cls, _ in comparisons:
            unique_classifications.add(source_cls)
            unique_classifications.add(target_cls)

        if progress_callback:
            progress_callback({
                "phase": "start",
                "message": "正在加载存储数据...",
                "progress": 0.0,
                "completed": 0,
                "total": total_comparisons,
            })

        chunk_data_map: Dict[str, ChunkData] = {}
        for cls in unique_classifications:
            # 协作式取消检查
            if cancel_event and cancel_event.is_set():
                return {
                    "status": "cancelled",
                    "comparisons": {},
                    "message": "去重任务在加载数据阶段被用户取消",
                }

            # 优先从 ChromaDB 读取存储向量
            data = self._load_chunk_data_from_chromadb(cls)
            if data is None:
                # Fallback: 重新嵌入
                data = self._load_chunk_data_by_embedding(cls)
            if data is not None:
                chunk_data_map[cls] = data

        if progress_callback:
            using_stored = all(cls in chunk_data_map for cls in unique_classifications)
            progress_callback({
                "phase": "start",
                "message": f"开始去重比对, 共需比对 {total_comparisons} 条数据",
                "progress": 0.0,
                "completed": 0,
                "total": total_comparisons,
            })

        results = {}
        completed_before = 0
        for i, (source_cls, target_cls, label) in enumerate(comparisons):
            # 协作式取消检查
            if cancel_event and cancel_event.is_set():
                if progress_callback:
                    progress_callback({
                        "phase": "cancelled",
                        "message": f"去重任务在比对 {label} 前被用户取消",
                        "progress": completed_before / total_comparisons,
                        "completed": completed_before,
                        "total": total_comparisons,
                    })
                # 仍然写入已完成组的结果
                break

            source_data = chunk_data_map.get(source_cls)
            target_data = chunk_data_map.get(target_cls)

            if source_data is None or target_data is None:
                print(f"[DEDUP] Skipping {label}: no data loaded")
                continue

            if progress_callback:
                progress_callback({
                    "phase": "progress",
                    "message": f"开始比对: {label} (第{i+1}/{len(comparisons)}组)",
                    "progress": completed_before / total_comparisons,
                    "completed": completed_before,
                    "total": total_comparisons,
                })

            print(f"[DEDUP] Starting comparison: {label}...")
            result = self._compare_collections(
                source_data=source_data,
                target_data=target_data,
                source_cls=source_cls,
                target_cls=target_cls,
                max_samples=max_samples,
                progress_callback=progress_callback,
                completed_before=completed_before,
                total_comparisons=total_comparisons,
                pair_label=label,
                cancel_event=cancel_event,
            )
            results[label] = result
            completed_before += result.total_comparisons_in_pair

            # 如果子比对被取消，终止外层循环
            if getattr(result, 'cancelled', False):
                break

            if completed_before > total_comparisons:
                total_comparisons = completed_before

            if progress_callback:
                progress_callback({
                    "phase": "progress",
                    "message": f"完成比对: {label}, 匹配 {len(result.matches)} 条",
                    "progress": completed_before / total_comparisons * 0.95,
                    "completed": completed_before,
                    "total": total_comparisons,
                })

        # 写入 Markdown 文件
        if progress_callback:
            progress_callback({
                "phase": "progress",
                "message": "正在写入比对结果文件...",
                "progress": 0.95,
                "completed": total_comparisons,
                "total": total_comparisons,
            })

        output_dir = Path(settings.DATA_DIR).parent / "dedup_results"
        output_dir.mkdir(exist_ok=True)

        # 整个"清空 → 写新文件"必须在锁内,否则并发 dedup 会出现:
        # T1 清空 → T2 清空(无事可做)→ T1 写文件 → T2 写文件 覆盖 → T1 看不到 T2 的输出
        with self._write_lock:
            # 清除之前遗留的结果文件，确保只保留最后一次比对结果
            for old_file in output_dir.glob("*.md"):
                old_file.unlink()
                print(f"[DEDUP] Removed old result: {old_file}")
            for old_file in output_dir.glob("*.json"):
                old_file.unlink()
                print(f"[DEDUP] Removed old result: {old_file}")

            for label, result in results.items():
                self._write_markdown(output_dir, label, result)
                self._write_json(output_dir, label, result)

            # 写入"准备去重.md": 超过 dedup_threshold 的匹配记录
            # ★ 必须在 _write_lock 内: 否则并发 dedup 时 T1 的 .md/.json 被 T2 清空覆盖后,
            # T1 仍基于自身 results 写出 准备去重.md, 与 .json(已是 T2)状态脱节。
            # ★ 使用 adjusted_similarity 与阈值比较, 与三档分桶一致
            # (dedup_threshold 是 adjusted 尺度, raw similarity 不可直接与 adjusted 阈值比较;
            #  注意 dedup 用线性 adjust, DocScan 用 √ adjust, 数值不等价)
            # B1: 同时要求 n-gram 字面支持 (overlap≥0.3 或 run≥30), 防止纯向量
            # boilerplate / 同领域不同内容被自动列入"准备去重"
            dedup_items = []
            for label, result in results.items():
                for m in result.matches:
                    if m["adjusted_similarity"] >= dedup_threshold:
                        text_overlap = m.get("text_overlap", 0.0)
                        longest_run = m.get("longest_run", 0)
                        if text_overlap < 0.3 and longest_run < 30:
                            continue  # n-gram 不支持, 跳过 (避免假阳性误删)
                        dedup_items.append({
                            "pair_label": label,
                            "pair_name": result.pair_name,
                            "source_chunk_id": m.get("source_chunk_id", ""),
                            "similarity": m["similarity"],
                            "adjusted_similarity": m["adjusted_similarity"],
                            "source_text": m["source_text"],
                            "target_text": m["target_text"],
                            "source_parent_text": m.get("source_parent_text", ""),
                            "target_parent_text": m.get("target_parent_text", ""),
                        })

            self._write_dedup_ready_md(output_dir, dedup_threshold, dedup_items)

        # 自动去重: 如果勾选了 auto_dedup，从高密级库删除超过阈值的 chunk
        dedup_deleted_total = 0
        if auto_dedup and dedup_items:
            dedup_deleted_total = self._auto_dedup_from_collections(dedup_items)

        overall_elapsed = time.time() - overall_start
        was_cancelled = cancel_event and cancel_event.is_set()

        return {
            "status": "cancelled" if was_cancelled else "completed",
            "comparisons": {
                label: {
                    "pair_name": result.pair_name,
                    "high_similarity": result.high_similarity_count,
                    "medium_similarity": result.medium_similarity_count,
                    "low_similarity": result.low_similarity_count,
                    "total_matches": len(result.matches),
                    "total_checked": result.total_checked,
                }
                for label, result in results.items()
            },
            "dedup_threshold": dedup_threshold,
            "dedup_ready_count": len(dedup_items),
            "dedup_deleted_count": dedup_deleted_total,
            "output_dir": str(output_dir),
            "elapsed_seconds": overall_elapsed,
        }

    @staticmethod
    def _atomic_write(filepath: Path, content: str) -> None:
        """原子写文本: 先写 .tmp 再 os.replace, 避免崩溃/并发中途留下半截文件.

        run_deduplication 在 _write_lock 内删除并重写各 *.json, 而
        apply_dedup_from_results 会读取它们; 非原子 write_text 有概率让读到
        截断 JSON (json.loads 失败被静默跳过 -> 去重数据丢失)。
        与 docscan._atomic_write_json 思路一致。
        """
        tmp_path = filepath.with_suffix(filepath.suffix + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, filepath)

    def _write_markdown(self, output_dir: Path, label: str, result: DedupResult):
        """将比对结果写入 Markdown 文件."""
        filename = f"{label}.md"
        filepath = output_dir / filename

        lines = [
            f"# {label} 数据去重比对结果",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"比对对: {result.pair_name}",
            f"比对总数: {result.total_checked}",
            f"高度相似(≥0.8): {result.high_similarity_count}",
            f"中度相似(0.65-0.8): {result.medium_similarity_count}",
            f"弱相关(0.5-0.65): {result.low_similarity_count}",
            f"总匹配数: {len(result.matches)}",
            "",
            "---",
            "",
            "## 详细匹配结果",
            "",
        ]

        for i, match in enumerate(result.matches, 1):
            lines.extend([
                f"### 匹配 #{i}",
                "",
                f"**相似度:** {match['similarity']:.4f}",
                "",
                "**源文档(子块):**",
                "```",
                match["source_text"],
                "```",
            ])
            if match.get("source_parent_text"):
                lines.extend([
                    "",
                    "**源文档(父块上下文):**",
                    "```",
                    match["source_parent_text"],
                    "```",
                ])
            lines.extend([
                "",
                "**目标文档(子块):**",
                "```",
                match["target_text"],
                "```",
            ])
            if match.get("target_parent_text"):
                lines.extend([
                    "",
                    "**目标文档(父块上下文):**",
                    "```",
                    match["target_parent_text"],
                    "```",
                ])
            lines.extend([
                "",
                "---",
                "",
            ])

        self._atomic_write(filepath, "\n".join(lines))
        print(f"[DEDUP] Written: {filepath}")

    def _write_json(self, output_dir: Path, label: str, result: DedupResult):
        """将比对结果写入 JSON 文件，供删除操作使用（包含 chunk ID）."""
        import json
        filename = f"{label}.json"
        filepath = output_dir / filename

        data = {
            "pair_name": result.pair_name,
            "total_checked": result.total_checked,
            "high_similarity_count": result.high_similarity_count,
            "medium_similarity_count": result.medium_similarity_count,
            "low_similarity_count": result.low_similarity_count,
            "total_matches": len(result.matches),
            "matches": [
                {
                    "source_chunk_id": m.get("source_chunk_id", ""),
                    "similarity": m["similarity"],
                    "adjusted_similarity": m["adjusted_similarity"],
                    "text_overlap": m.get("text_overlap", 0.0),
                    "longest_run": m.get("longest_run", 0),
                    "source_text": m["source_text"],
                    "target_text": m["target_text"],
                    "source_parent_text": m.get("source_parent_text", ""),
                    "target_parent_text": m.get("target_parent_text", ""),
                }
                for m in result.matches
            ],
        }

        self._atomic_write(filepath, json.dumps(data, ensure_ascii=False, indent=2))
        print(f"[DEDUP] Written: {filepath}")

    def _write_dedup_ready_md(self, output_dir: Path, threshold: float, items: list):
        """写入"准备去重.md": 超过阈值的匹配记录汇总."""
        filepath = output_dir / "准备去重.md"

        lines = [
            f"# 准备去重数据 (相似度 ≥ {threshold})",
            "",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"去重阈值: {threshold}",
            f"待去重数据条数: {len(items)}",
            "",
            "---",
            "",
        ]

        # 按比对组分类
        from itertools import groupby
        items_sorted = sorted(items, key=lambda x: x["pair_label"])
        for pair_label, group in groupby(items_sorted, key=lambda x: x["pair_label"]):
            group_list = list(group)
            lines.append(f"## {pair_label} ({len(group_list)} 条)")
            lines.append("")
            for i, item in enumerate(group_list, 1):
                lines.extend([
                    f"### #{i}  相似度: {item['similarity']:.4f}",
                    "",
                    f"**源文档 chunk ID:** {item['source_chunk_id']}",
                    "",
                    "**源文档(子块):**",
                    "```",
                    item["source_text"],
                    "```",
                ])
                if item.get("source_parent_text"):
                    lines.extend([
                        "",
                        "**源文档(父块上下文):**",
                        "```",
                        item["source_parent_text"],
                        "```",
                    ])
                lines.extend([
                    "",
                    "**目标文档(子块):**",
                    "```",
                    item["target_text"],
                    "```",
                ])
                if item.get("target_parent_text"):
                    lines.extend([
                        "",
                        "**目标文档(父块上下文):**",
                        "```",
                        item["target_parent_text"],
                        "```",
                    ])
                lines.extend([
                    "",
                    "---",
                    "",
                ])

        self._atomic_write(filepath, "\n".join(lines))
        print(f"[DEDUP] Written dedup ready file: {filepath} ({len(items)} items)")

    def _auto_dedup_from_collections(self, items: list) -> int:
        """自动去重: 从高密级库删除超过阈值的 chunk.

        Args:
            items: 超过阈值的匹配记录列表, 每项含 source_chunk_id

        Returns:
            int: 总删除数量
        ★ 复用 vector_db_service 的共享 ChromaDB client,避免多客户端锁竞争。
        """

        # 按源分类分组 chunk IDs
        LABEL_TO_CLASSIFICATION = {
            "机密-受限": ("confidential", "restricted"),
            "机密-公开": ("confidential", "public"),
            "受限-公开": ("restricted", "public"),
        }

        # 收集每个 source_cls 要删除的 chunk IDs
        ids_by_cls: Dict[str, set] = {}
        for item in items:
            pair = LABEL_TO_CLASSIFICATION.get(item["pair_label"])
            if not pair:
                continue
            source_cls = pair[0]
            chunk_id = item.get("source_chunk_id")
            if chunk_id:
                ids_by_cls.setdefault(source_cls, set()).add(chunk_id)

        # Backup deleted IDs before actual deletion
        backup_path = Path(settings.DATA_DIR).parent / "dedup_results" / f"dedup_deletion_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_data = {"deleted_ids": {cls: list(ids) for cls, ids in ids_by_cls.items()}, "timestamp": datetime.now().isoformat()}
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        # P2-11: 显式 utf-8 + ensure_ascii=False, 避免中文 Windows 默认 GBK 写出
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)

        total_deleted = 0
        client = vector_db_service.get_client()

        for source_cls, chunk_ids in ids_by_cls.items():
            child_collection_name = f"md2rag_{source_cls}_child"
            try:
                collection = client.get_collection(child_collection_name)
                collection.delete(ids=list(chunk_ids))
                total_deleted += len(chunk_ids)
                print(f"[DEDUP-AUTO] Deleted {len(chunk_ids)} chunks from {child_collection_name}")
            except Exception as e:
                print(f"[DEDUP-AUTO] Error deleting from {child_collection_name}: {e}")

        return total_deleted

    def apply_dedup_from_results(self, dedup_threshold: float) -> Dict[str, Any]:
        """按照比对结果文件去重: 从高密级库删除相似度超过阈值的 chunk.

        读取各组的 JSON 结果文件, 提取超过阈值的 source_chunk_id,
        从对应的高密级 child collection 中删除.
        操作只在向量数据库中进行, 不涉及 md 和原始文件.

        Args:
            dedup_threshold: 去重相似度阈值

        Returns:
            Dict: 删除结果统计
        """
        import json

        LABEL_TO_CLASSIFICATION = {
            "机密-受限": ("confidential", "restricted"),
            "机密-公开": ("confidential", "public"),
            "受限-公开": ("restricted", "public"),
        }

        dedup_dir = Path(settings.DATA_DIR).parent / "dedup_results"

        # 收集每个 source_cls 要删除的 chunk IDs
        ids_by_cls: Dict[str, set] = {}
        total_items = 0

        # ★ 与 run_deduplication 互斥: 后者在 _write_lock 内删除并重写这些 json
        # (现已原子写), 不加锁会出现 apply 读到被删/被截断的 json, 进而基于过期
        # 结果删 chunk。读取临界区在锁内, 读到的 ids 与落盘结果一致。
        with self._write_lock:
            for pair_label, (source_cls, _) in LABEL_TO_CLASSIFICATION.items():
                json_file = dedup_dir / f"{pair_label}.json"
                if not json_file.exists():
                    continue

                try:
                    result_data = json.loads(json_file.read_text(encoding="utf-8"))
                except Exception:
                    continue

                for m in result_data.get("matches", []):
                    # ★ 使用 adjusted_similarity 与阈值比较, 与比对筛选和三档分桶一致
                    # (dedup_threshold 是 adjusted 尺度, raw similarity 不能直接比较)
                    if "adjusted_similarity" in m:
                        adj_sim = m["adjusted_similarity"]
                    elif "similarity" in m:
                        # 旧版/外部结果缺 adjusted_similarity: 用线性公式从 raw 换算
                        # (与 run_deduplication 的 adjust 一致), 而非直接拿 raw 比阈值
                        # (raw 尺度偏高, 会把不达标的对误判为重复)
                        raw = m["similarity"]
                        adj_sim = max(0.0, (raw - BGE_M3_BASELINE) / (1.0 - BGE_M3_BASELINE))
                    else:
                        adj_sim = 0.0
                    if adj_sim >= dedup_threshold and m.get("source_chunk_id"):
                        ids_by_cls.setdefault(source_cls, set()).add(m["source_chunk_id"])
                        total_items += 1

        if total_items == 0:
            return {
                "status": "skipped",
                "message": f"在比对结果中未找到相似度 ≥ {dedup_threshold} 的匹配数据",
                "deleted_count": 0,
            }

        # 执行删除 — ★ 复用 vector_db_service 的共享 ChromaDB client
        client = vector_db_service.get_client()

        # Backup deleted IDs before actual deletion
        backup_path = Path(settings.DATA_DIR).parent / "dedup_results" / f"dedup_apply_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_data = {"deleted_ids": {cls: list(ids) for cls, ids in ids_by_cls.items()}, "threshold": dedup_threshold, "timestamp": datetime.now().isoformat()}
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        # P2-11: 显式 utf-8 + ensure_ascii=False, 避免中文 Windows 默认 GBK 写出
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)

        total_deleted = 0
        details = {}

        for source_cls, chunk_ids in ids_by_cls.items():
            child_collection_name = f"md2rag_{source_cls}_child"
            try:
                collection = client.get_collection(child_collection_name)
                collection.delete(ids=list(chunk_ids))
                total_deleted += len(chunk_ids)
                details[source_cls] = len(chunk_ids)
                print(f"[DEDUP-APPLY] Deleted {len(chunk_ids)} chunks from {child_collection_name}")
            except Exception as e:
                print(f"[DEDUP-APPLY] Error deleting from {child_collection_name}: {e}")

        return {
            "status": "success",
            "message": f"已从高密级库删除 {total_deleted} 条相似度 ≥ {dedup_threshold} 的数据 (涉及 {len(ids_by_cls)} 个密级: {details})",
            "deleted_count": total_deleted,
            "threshold": dedup_threshold,
            "details": details,
        }


deduplication_service = DataDeduplicationService()
