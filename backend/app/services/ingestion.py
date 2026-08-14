"""基于 MD2RAG 的入库服务 - 替代简陋的 backend ingestion.py.

薄包装: 调用 MD2RAG Indexer 暴露数据入库能力
- 父子块双 collection 入库
- bbox 元数据追踪
- LLM 摘要 (X2MD 已生成 abstract, 可选 LLM 重新生成)
- 图片入库 (ViT-Large)
- 维度变化时自动 reset

V2 优化:
- 分批入库，避免内存溢出
- 流式处理，及时释放内存
- 细粒度进度回调
"""

from __future__ import annotations

import gc
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# MD2RAG path is configured in app/main.py at startup; no need to manipulate sys.path here.

from app.core.config import settings
from app.services.vector_db import vector_db_service
from md2rag.loader import CLASSIFICATION_DIR_MAP


def create_embedder_from_settings():
    """根据 backend settings 创建 embedder, 判序与 Indexer._create_embedder 一致.

    scanner fallback 与 indexer 共用此函数, 避免入库向量与查询向量空间不一致
    (旧 scanner fallback 自行按 EMBEDDING_MODEL 字符串建 embedder, 缺 ollama_cache_path
    且判序与 indexer 不同)。用完必须由调用方 release()。

    判序 (与 _build_md2rag_config 对齐, backend settings 不暴露 sentence-transformers):
      ollama-bge-m3 -> ollama (含 embedding cache)
      mps-bge-m3    -> mps
      其它          -> chromadb-default
    """
    from md2rag.embedder import create_embedder

    model = (settings.EMBEDDING_MODEL or "").lower()
    if model == "ollama-bge-m3":
        return create_embedder(
            model_type="ollama",
            ollama_url=settings.OLLAMA_BASE_URL,
            ollama_model="bge-m3:latest",
            ollama_cache_path=str(Path(settings.VECTOR_DB_DIR) / "md2rag_embedding_cache.sqlite"),
        )
    if model == "mps-bge-m3":
        return create_embedder(model_type="mps", device="mps")
    return create_embedder(model_type="chromadb-default")


class DataIngestionService:
    """入库服务 - 薄包装 MD2RAG Indexer (V2 内存优化版)."""

    def __init__(self):
        self._indexer = None
        self._md2rag_config = None
        # 防 reset 并发 (双击 / 清空+切换模型 同时触发): 非阻塞获取,
        # 抢不到说明已有 reset 在跑, 直接返回 status="busy" 而非报错/partial。
        self._reset_lock = threading.Lock()

    def _get_indexer(self):
        """延迟初始化 MD2RAG Indexer (原 V2 分批版本, 已合并为唯一实现)."""
        if self._indexer is not None:
            return self._indexer

        # indexer.py 已是合并后的分批流式版本 (原 indexer_v2), 不再有 V1/V2 分支
        from md2rag.indexer import Indexer
        cfg = self._build_md2rag_config()
        self._md2rag_config = cfg
        self._indexer = Indexer(cfg)
        return self._indexer

    def _build_md2rag_config(self):
        """构建 MD2RAG 配置对象 — V1/V2 合并后的单一 Indexer 使用."""
        from md2rag.config import load_config as md2rag_load_config

        cfg = md2rag_load_config()
        # 覆盖配置：使用 backend 的 VECTOR_DB_DIR
        cfg.vector_db_dir = Path(settings.VECTOR_DB_DIR)
        cfg.md_dir = Path(settings.DATA_DIR)
        cfg.collection_prefix = "md2rag"
        # 嵌入器分支: ollama-bge-m3 (默认) | mps-bge-m3 | 其它本地模型
        cfg.embedding_model = settings.EMBEDDING_MODEL
        if settings.EMBEDDING_MODEL == "ollama-bge-m3":
            cfg.ollama_enabled = True
            cfg.ollama_model = "bge-m3:latest"
            cfg.device = "auto"
        elif settings.EMBEDDING_MODEL == "mps-bge-m3":
            cfg.ollama_enabled = False
            cfg.st_enabled = False
            cfg.device = "mps"
        else:
            cfg.ollama_enabled = False
            cfg.device = settings.DEVICE if hasattr(settings, 'DEVICE') else "auto"

        return cfg

    def estimate_records(self, classification_level: str) -> Dict[str, Any]:
        """预估入库时间和待入库文档数.

        通过 MD2RAG 的 ChunkLoader 发现切片文件, 按文档 (切片 json 内的
        metadata.source, 即入库写入 ChromaDB 的同一字段) 去重, 并排除已入库
        文档后返回 "待入库" 数量.

        file_count 语义为 "待入库文档数" (已排除已入库), 不是切片文件数:
        父子块模式下每个文档贡献 parents+children 两个切片文件, 直接数切片文件
        会把文档数翻倍, 故改为按 source 聚合.
        """
        try:
            from md2rag.loader import CLASSIFICATION_DIR_MAP, ChunkLoader

            dir_name = CLASSIFICATION_DIR_MAP.get(classification_level)
            if not dir_name:
                return {"error": "Invalid classification level"}

            dir_path = os.path.join(settings.DATA_DIR, dir_name)
            if not os.path.exists(dir_path):
                return {
                    "status": "no_directory",
                    "classification_level": classification_level,
                    "estimated_records": 0,
                    "file_count": 0,
                    "total_documents": 0,
                    "indexed_documents": 0,
                    "pending_documents": 0,
                    "estimated_time_seconds": 0,
                    "message": f"Directory does not exist: {dir_path}",
                }

            loader = ChunkLoader(settings.DATA_DIR)
            files = loader.discover_files(classification_level)

            indexed_sources = self._get_indexed_sources(classification_level)

            doc_sources: set = set()
            src_sizes: Dict[str, int] = {}
            files_info = []
            for f in files:
                size = f.stat().st_size
                src = self._source_from_path(f)
                if src:
                    doc_sources.add(src)
                    src_sizes[src] = src_sizes.get(src, 0) + size
                files_info.append({
                    "name": f.name,
                    "path": str(f),
                    "size": size,
                    "type": f.suffixes[-2] if len(f.suffixes) > 1 else "",
                })

            total_documents = len(doc_sources)
            pending_sources = doc_sources - indexed_sources
            indexed_documents = total_documents - len(pending_sources)
            # 粗略估算: 1000 字节约 1 chunk; 只累计待入库文档, 避免把已入库也算进去
            estimated_records = sum(
                max(1, sz // 1000)
                for src, sz in src_sizes.items()
                if src in pending_sources
            )
            estimated_time = estimated_records * 0.1 + len(pending_sources) * 0.5
            return {
                "status": "estimated",
                "classification_level": classification_level,
                "estimated_records": estimated_records,
                "file_count": len(pending_sources),
                "total_documents": total_documents,
                "indexed_documents": indexed_documents,
                "pending_documents": len(pending_sources),
                "estimated_time_seconds": estimated_time,
                "estimated_time_formatted": self._format_time(estimated_time),
                "files": files_info,
                "message": (
                    f"待入库文档: {len(pending_sources)}/{total_documents}, "
                    f"切片文件: {len(files_info)}"
                ),
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def _source_from_path(file_path) -> Optional[str]:
        """从切片文件名推导文档 source (与入库 metadata.source 同口径).

        切片文件命名为 <source>.{parents,children,chunks}.json, 其中 <source>
        即入库写入 ChromaDB 的 source 字段 (parents/chunks/children 共享同一值).
        从文件名反推可避免逐个 json.load —— 大库 (2.7w+ 切片文件) 读一遍要 ~80s,
        不可接受; 文件名推导与 stat 同级开销.
        """
        name = Path(file_path).name
        for suf in (".parents.json", ".children.json", ".chunks.json"):
            if name.endswith(suf):
                return name[:-len(suf)]
        return None

    def _collect_doc_sources(self, files) -> set:
        """从切片文件聚合文档身份 (source) 集合, 用于入库去重预检."""
        sources: set = set()
        for f in files:
            # 修正: 此前误调不存在的 self._read_source -> AttributeError 被外层
            # except 吞掉, 去重预检恒失败退化为全量重嵌, "全部已入库则跳过"永不触发
            src = self._source_from_path(f)
            if src:
                sources.add(src)
        return sources

    def _get_indexed_sources(self, classification_level: str) -> set:
        """查 SQLite 拿该密级下已索引的 source 文件路径集合.

        用于 ingest_directory 的去重预检 - 已入库的文件不再走 MD2RAG embedder.
        通过 chunk metadata 中的 'source' 字段聚合 (每个 chunk 都带原文件路径).
        """
        import sqlite3
        db_path = Path(settings.VECTOR_DB_DIR) / "chroma.sqlite3"
        if not db_path.exists():
            return set()
        sources: set = set()
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                cursor = conn.cursor()
                # 找该 classification 下所有 chunk 的 source 字段
                # embedding_metadata: id (embedding 主键) + key + string_value
                cursor.execute(
                    """
                    SELECT DISTINCT em2.string_value
                    FROM embedding_metadata em1
                    JOIN embedding_metadata em2 ON em1.id = em2.id
                    WHERE em1.key = 'classification' AND em1.string_value = ?
                      AND em2.key = 'source' AND em2.string_value IS NOT NULL
                    """,
                    (classification_level,),
                )
                for row in cursor.fetchall():
                    if row[0]:
                        sources.add(row[0])
            finally:
                conn.close()
        except Exception as e:
            print(f"[INGEST] _get_indexed_sources failed (treat as empty): {e}")
        return sources

    def ingest_directory(
        self,
        classification_level: str,
        progress_callback: Optional[Callable] = None,
        strategy: str = "auto",
        include_images: bool = True,
        force: bool = False,
        cancel_event=None,
        release_after: bool = True,
    ) -> Dict:
        """入库指定密级.

        Args:
            classification_level: public / restricted / confidential
            strategy: 切片策略 (auto/chunk/parent-child)
            include_images: 是否同时处理图片 (ViT-Large)
            force: True 时跳过去重预检, 强制重新 embed 所有文件
                   (ChromaDB ID 冲突仍由 MD2RAG try/except 吞掉, 但会浪费 embed 时间)

        Returns:
            Dict: 入库结果. 全部已入库时 status='skipped', 此时不调 MD2RAG.
        """
        try:
            # 去重预检 (P2-3): 已入库文档不再 embed
            if not force:
                try:
                    from md2rag.loader import ChunkLoader
                    loader = ChunkLoader(settings.DATA_DIR)
                    discovered = loader.discover_files(classification_level) or []
                    # 用切片 json 内的 metadata.source 聚合文档身份 (与入库写入
                    # ChromaDB 的 source 字段同口径); 此前用 str(path) 完整路径与
                    # _get_indexed_sources 返回的裸名对不上, 交集恒空 -> 短路永不触发.
                    discovered_sources = self._collect_doc_sources(discovered)
                    indexed_sources = self._get_indexed_sources(classification_level)
                    new_sources = discovered_sources - indexed_sources
                    overlapping = discovered_sources & indexed_sources

                    if discovered_sources and not new_sources:
                        # 100% 已入库, 短路返回, 不浪费 embed 时间
                        msg = (
                            f"全部 {len(discovered_sources)} 个文档均已入库, "
                            f"跳过 (传 force=True 可强制重入)"
                        )
                        if progress_callback:
                            progress_callback({
                                "phase": "skipped",
                                "level": classification_level,
                                "message": msg,
                            })
                        return {
                            "status": "skipped",
                            "message": msg,
                            "documents_processed": 0,
                            "chunks_added": 0,
                            "parents_added": 0,
                            "children_added": 0,
                            "images_added": 0,
                            "classification_level": classification_level,
                            "time_elapsed_seconds": 0,
                            "errors": [],
                            "skipped_existing": len(indexed_sources),
                            "discovered_total": len(discovered_sources),
                        }

                    if overlapping:
                        # 部分新增、部分已入: 仍跑 MD2RAG, 由 ChromaDB ID 冲突吞掉旧的
                        # (理想方案是只传 new_sources 给 MD2RAG, 但需改其私有 API)
                        print(
                            f"[INGEST] {classification_level}: {len(new_sources)} 新增文档, "
                            f"{len(overlapping)} 已存在 (将由 ChromaDB ID 冲突跳过)"
                        )
                except Exception as e:
                    # 预检失败不能阻塞主流程, 退化为旧行为
                    print(f"[INGEST] dedup precheck failed (fallback to full ingest): {e}")

            indexer = self._get_indexer()

            # 协作式取消检查: 在启动耗时的索引操作前检查
            if cancel_event and cancel_event.is_set():
                if progress_callback:
                    progress_callback({
                        "phase": "cancelled",
                        "level": classification_level,
                        "message": f"{classification_level} 入库被用户取消",
                    })
                return {
                    "status": "cancelled",
                    "message": f"{classification_level} 入库被用户取消",
                    "documents_processed": 0,
                    "chunks_added": 0,
                    "parents_added": 0,
                    "children_added": 0,
                    "images_added": 0,
                    "classification_level": classification_level,
                }

            if progress_callback:
                progress_callback({
                    "phase": "start",
                    "level": classification_level,
                    "message": f"开始 MD2RAG 索引 {classification_level}",
                })

            result = indexer.index_directory(
                classification=classification_level,
                strategy=strategy,
                include_images=include_images,
                progress_callback=progress_callback,
                cancel_event=cancel_event,  # 传入取消信号,支持中途取消
            )

            if progress_callback:
                progress_callback({
                    "phase": "done",
                    "level": classification_level,
                    "documents": result.documents_processed,
                    "parents": result.parents_added,
                    "children": result.children_added,
                    "images": result.images_added,
                })

            return {
                "status": result.status,
                "message": result.message,
                "documents_processed": result.documents_processed,
                "chunks_added": result.parents_added + result.children_added,
                "parents_added": result.parents_added,
                "children_added": result.children_added,
                "images_added": result.images_added,
                "classification_level": classification_level,
                "time_elapsed_seconds": 0,
                "errors": result.errors,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "documents_processed": 0,
                "chunks_added": 0,
                "parents_added": 0,
                "children_added": 0,
                "images_added": 0,
            }
        finally:
            # 入库结束，释放 Indexer 及底层模型/Chroma 引用，降低常驻内存
            # 仅在独立调用时释放; 由 ingest_all_levels 调用时保留 indexer 以避免重复加载
            if release_after:
                self._cleanup_memory()

    def ingest_all_levels(
        self,
        progress_callback: Optional[Callable] = None,
        force: bool = False,
        strategy: str = "auto",
        include_images: bool = True,
        cancel_event=None,
    ) -> Dict:
        """入库所有密级（public / restricted / confidential）.

        strategy / include_images 透传给 ingest_directory,与单密级模式行为对齐;
        旧版本签名缺这两个参数会让 UI 的"跳过图片/切片策略"开关在多密级入库时失效。

        ★ 内存策略 (2026-06-24): 每个密级跑完都强制释放 indexer (release_after=True),
        旧实现传 False 为了"避免重复加载模型", 但 bge-m3 MPS 模型 + 各密级 collection
        累积导致进程 RSS 在 ingest 中段就 OOM。重加载模型耗时几秒, 远低于 OOM 代价。
        """
        overall_start = time.time()
        results = {}
        for level in ["public", "restricted", "confidential"]:
            # 协作式取消检查
            if cancel_event and cancel_event.is_set():
                if progress_callback:
                    progress_callback({
                        "phase": "cancelled",
                        "level": level,
                        "message": f"{level} 入库前被用户取消",
                    })
                break

            if progress_callback:
                progress_callback({
                    "phase": "start",
                    "level": level,
                    "message": f"开始导入 {level} 级别文档",
                })

            # Capture level by value to avoid the classic Python loop-closure bug
            # (deferred callbacks would otherwise see the final loop value)
            def make_level_progress(current_level):
                def _level_progress(data):
                    if progress_callback:
                        data["level"] = current_level
                        progress_callback(data)
                return _level_progress

            results[level] = self.ingest_directory(
                level, make_level_progress(level),
                strategy=strategy, include_images=include_images,
                force=force, cancel_event=cancel_event, release_after=True,
            )

        total_time = time.time() - overall_start
        total_docs = sum(r.get("documents_processed", 0) for r in results.values())
        total_chunks = sum(r.get("chunks_added", 0) for r in results.values())
        total_images = sum(r.get("images_added", 0) for r in results.values())
        was_cancelled = cancel_event and cancel_event.is_set()
        # 兜底再清一次 (单密级 finally 已清, 这里防御性回收)
        self._cleanup_memory()
        return {
            "status": "cancelled" if was_cancelled else "completed",
            "total_documents": total_docs,
            "chunks_added": total_chunks,
            "images_added": total_images,
            "total_time_seconds": total_time,
            "total_time_formatted": self._format_time(total_time),
            "message": f"全部入库完成: {total_chunks} 个文本切片 + {total_images} 张图片",
            "details": results,
        }

    def get_database_stats(self) -> Dict[str, int]:
        """获取向量数据库统计.

        直接从 SQLite 数据库查询，避免 MD2RAG collection 名称不匹配问题.
        """
        import sqlite3

        try:
            db_path = Path(settings.VECTOR_DB_DIR) / "chroma.sqlite3"
            if not db_path.exists():
                return {
                    "collections": {
                        "public_documents": 0,
                        "restricted_documents": 0,
                        "confidential_documents": 0,
                    },
                    "total_vectors": 0,
                }

            conn = sqlite3.connect(str(db_path))
            try:
                cursor = conn.cursor()

                # 获取所有 collection 名称和 ID
                cursor.execute("SELECT id, name FROM collections;")
                collections = {row[1]: row[0] for row in cursor.fetchall()}

                # 统计各密级的 embeddings 数量
                # 通过 embedding_metadata 中的 classification 字段统计
                # (跨所有 collection 聚合,与具体 collection 命名解耦)
                stats = {cls: 0 for cls in CLASSIFICATION_DIR_MAP}

                for classification in CLASSIFICATION_DIR_MAP:
                    cursor.execute(
                        """
                        SELECT COUNT(DISTINCT e.id)
                        FROM embeddings e
                        JOIN embedding_metadata em ON e.id = em.id
                        WHERE em.key = 'classification' AND em.string_value = ?
                        """,
                        (classification,),
                    )
                    result = cursor.fetchone()
                    if result:
                        stats[classification] = result[0]
            finally:
                # 异常路径也必须关闭连接, 否则反复失败会泄漏 sqlite 句柄
                # (对比 _get_indexed_sources 已用 try/finally)
                conn.close()

            # 注: collections key 仍沿用 *_documents 名以兼容前端,但实际数据
            # 已统一从 md2rag_{cls}_* collection 聚合(由 classification metadata 区分)
            return {
                "collections": {
                    "public_documents": stats["public"],
                    "restricted_documents": stats["restricted"],
                    "confidential_documents": stats["confidential"],
                },
                "total_vectors": stats["public"] + stats["restricted"] + stats["confidential"],
            }
        except Exception:
            # 降级：尝试通过 MD2RAG API 获取
            try:
                indexer = self._get_indexer()
                stats = indexer.get_stats()
                return {
                    "collections": {
                        "public_documents": stats.get("public_parents", 0) + stats.get("public_children", 0),
                        "restricted_documents": stats.get("restricted_parents", 0) + stats.get("restricted_children", 0),
                        "confidential_documents": stats.get("confidential_parents", 0) + stats.get("confidential_children", 0),
                    },
                    "total_vectors": (
                        stats.get("public", 0)
                        + stats.get("restricted", 0)
                        + stats.get("confidential", 0)
                    ),
                }
            except Exception:
                return {
                    "collections": {
                        "public_documents": 0,
                        "restricted_documents": 0,
                        "confidential_documents": 0,
                    },
                    "total_vectors": 0,
                }

    def switch_embedding_model(self, new_model: str) -> Dict[str, Any]:
        """切换嵌入模型并保证两套 embedder 与向量库一致.

        切模型时向量维度/语义改变, 必须: 持久化新模型 + 清空旧向量 + 释放旧 indexer
        (下次 _get_indexer 按新模型重建) + 重建 backend 兜底 embedding_fn.
        缺任一步都会导致入库或检索仍用旧模型, 维度混用致结果错乱.

        此前 start_ingest_task 仅做 settings.EMBEDDING_MODEL = new_model 内存赋值,
        不持久化 (重启丢) / 不清库 / 不重建 embedder, 是模型切换 bug 的根因.
        """
        from app.services.settings_service import settings_service

        # 1) 持久化 + 应用到 settings (settings_service 校验白名单并 _apply_runtime_settings)
        settings_service.update_category("model", {"embeddingModel": new_model})

        # 2) 清空 MD2RAG collection + 释放旧 indexer (下次 _get_indexer 按新模型重建)
        res = self.reset_all()
        if res.get("status") in ("error", "busy"):
            raise RuntimeError(f"切换模型清库失败: {res.get('message')}")

        # 3) 重建 backend 兜底 embedding_fn + 清失效 collection 缓存 (不重复删数据)
        vector_db_service.rebuild_embedding_fn()
        return {"switched_to": new_model, "reset": res}

    def reset_all(self):
        """重置所有 collection + 清理物理残留.

        1) 互斥: 同一时刻只允许一个 reset (防双击/多路径并发 -> 竞态误报 error/partial)
        2) 记 before, 调 indexer.clear() 全清
        3) 兜底: 枚举删除残留 collection
        4) 物理清理: 删除 ChromaDB HNSW UUID 目录 + SQLite VACUUM
        5) 以清理后的最终计数判定 success/partial (中途 after 在并发/事务延迟下可能读到陈旧 >0)
        """
        # 互斥: 非阻塞获取。并发 reset (双击 / 清空+切换模型 同时触发) 直接返回 busy,
        # 不抛异常也不报 partial, 让前端按 status="busy" 友好提示, 避免假"失败"。
        if not self._reset_lock.acquire(blocking=False):
            return {
                "status": "busy",
                "message": "Reset already in progress, please wait",
                "vectors_before": None,
                "vectors_after": None,
            }
        try:
            before = self.get_database_stats().get("total_vectors", -1)
            indexer = self._get_indexer()
            indexer.clear()
            # 强制释放 indexer 持有的 client 引用, 否则 after 统计可能读到缓存
            self._release_indexer()
            after = self.get_database_stats().get("total_vectors", -1)
            if after > 0:
                # 兜底: indexer.clear 没清掉的剩余 collection 再扫一遍
                self._force_purge_remaining()
                after = self.get_database_stats().get("total_vectors", -1)

            # 物理清理: ChromaDB 的 delete_collection 只删 SQLite 记录,
            # 不删 HNSW 向量索引目录(UUID 子目录),也不回收 SQLite 空间。
            # 这些残留不影响功能但浪费磁盘, 在 reset 时一并清除。
            self._physical_cleanup()

            # 以物理清理完成后的最终计数为准: 此前的 after 在并发/事务提交延迟下
            # 可能读到陈旧 >0 (历史假"partial"的根因)。clear()+purge+cleanup 已尽力清空,
            # 这重查一次拿到确定结果。
            final = self.get_database_stats().get("total_vectors", -1)

            return {
                "status": "success" if final == 0 else "partial",
                "message": f"Reset: {before} -> {final} vectors",
                "vectors_before": before,
                "vectors_after": final,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            self._reset_lock.release()

    def _force_purge_remaining(self) -> None:
        """兜底: 用裸 ChromaDB client 枚举并删除所有残留 collection."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            client = chromadb.PersistentClient(
                path=str(settings.VECTOR_DB_DIR),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            for coll in client.list_collections():
                name = coll.name if hasattr(coll, "name") else str(coll)
                try:
                    client.delete_collection(name=name)
                    print(f"[RESET-PURGE] Deleted residual: {name}")
                except Exception as e:
                    print(f"[RESET-PURGE] Failed {name}: {e}")
        except Exception as e:
            print(f"[RESET-PURGE] Fallback failed: {e}")

    def _physical_cleanup(self) -> None:
        """清理 ChromaDB reset 后的物理残留: HNSW UUID 目录 + SQLite VACUUM.

        ChromaDB 的 delete_collection() 会删 SQLite 里的 collection/segment 记录,
        但不会删除磁盘上对应的 HNSW 索引目录(UUID 子目录)。
        SQLite 删除大量行后也不自动回收空间(需显式 VACUUM)。
        """
        db_dir = Path(settings.VECTOR_DB_DIR)
        sqlite_path = db_dir / "chroma.sqlite3"

        # 1. 删除残留的 UUID 目录(每个是已删 collection 的 HNSW 索引)
        #    UUID 目录的特征: 名为 hex UUID, 内含 data_level0.bin/header.bin 等
        import re
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
        cleaned_dirs = 0
        for entry in db_dir.iterdir():
            if entry.is_dir() and uuid_pattern.match(entry.name):
                # 确认是 HNSW 目录(含 header.bin),且不是符号链接(防 symlink 穿越)
                if (entry / "header.bin").exists() and not entry.is_symlink():
                    try:
                        import shutil
                        shutil.rmtree(entry)
                        cleaned_dirs += 1
                    except Exception as e:
                        print(f"[RESET-CLEANUP] Failed to remove {entry.name}: {e}")

        # 2. SQLite VACUUM 回收空间(需要无其他进程持有 SQLite 锁)
        if sqlite_path.exists() and cleaned_dirs >= 0:  # 即使无目录残留也做 VACUUM
            try:
                import sqlite3
                conn = sqlite3.connect(str(sqlite_path))
                try:
                    conn.execute("VACUUM")
                finally:
                    # VACUUM 抛 OperationalError("database is locked") 时也要关闭连接,
                    # 否则反复 reset 会泄漏 sqlite 句柄直至 fd 耗尽
                    conn.close()
                print(f"[RESET-CLEANUP] SQLite VACUUM done")
            except sqlite3.OperationalError as e:
                # 常见: "database is locked" — backend 进程仍持有连接, 无法 VACUUM
                # 不阻塞 reset 流程, 下次 stop+start 时可手动 VACUUM
                print(f"[RESET-CLEANUP] SQLite VACUUM skipped (locked or busy): {e}")
            except Exception as e:
                print(f"[RESET-CLEANUP] SQLite VACUUM failed: {e}")

        if cleaned_dirs > 0:
            print(f"[RESET-CLEANUP] Removed {cleaned_dirs} orphaned HNSW directories")

    def _release_indexer(self) -> None:
        """显式释放 Indexer 模型/缓存，降低进程常驻内存。"""
        try:
            if self._indexer is not None:
                self._indexer.release()
        except Exception:
            pass
        self._indexer = None
        self._md2rag_config = None
        gc.collect()

    def _cleanup_memory(self) -> None:
        """通用内存清理（GC + torch 缓存 + reranker 卸载）。

        reranker 在 DocScan compare 路径会加载 ~568M params 模型, 与 ingestion
        共进程时占据 ~1GB+. 入库完成后 reranker 暂时也用不到, 一并卸载。
        """
        self._release_indexer()
        try:
            from app.services.reranker import reranker
            reranker.release()
        except Exception:
            pass
        try:
            import torch
            torch.set_num_threads(1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
        except Exception:
            pass
        gc.collect()

    def cleanup_memory(self) -> Dict[str, str]:
        """对外暴露的内存清理入口。"""
        self._cleanup_memory()
        return {"status": "success", "message": "Ingestion memory cache released"}

    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}秒"
        if seconds < 3600:
            return f"{seconds / 60:.1f}分钟"
        return f"{seconds / 3600:.1f}小时"


data_ingestion_service = DataIngestionService()
