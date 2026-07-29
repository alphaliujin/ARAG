"""MD2RAG API 服务 - 提供完整的数据入库功能."""

import os
import threading
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from md2rag.config import MD2RAGConfig, load_config
from md2rag.indexer import Indexer
from md2rag.loader import CLASSIFICATION_DIR_MAP, VALID_CLASSIFICATIONS
from md2rag.logger import get_logger, log_step, log_timing

# 创建日志记录器
logger = get_logger("md2rag.api")

app = FastAPI(
    title="MD2RAG API",
    version="0.2.0",
    description="MD2RAG 数据入库服务"
)

# CORS 中间件 — 旧版用 ["*"] 在浏览器场景配合 cookies 时有 CSRF 风险;
# 这里由环境变量控制,默认仅允许同源 + 本地开发 (3000)。
_allowed_origins_env = os.environ.get("MD2RAG_CORS_ORIGINS", "").strip()
if _allowed_origins_env:
    allowed_origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
else:
    allowed_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# 全局配置和索引器
logger.info("=" * 80)
logger.info("MD2RAG API Server Starting...")
logger.info("=" * 80)

log_step(logger, "INIT", "Loading configuration...")
_config = load_config()
logger.info(f"[CONFIG] md_dir: {_config.md_dir}")
logger.info(f"[CONFIG] vector_db_dir: {_config.vector_db_dir}")
logger.info(f"[CONFIG] embedding_model: {_config.embedding_model}")

log_step(logger, "INIT", "Initializing Indexer...")
start_time = time.time()
_indexer = Indexer(_config)
init_elapsed = (time.time() - start_time) * 1000
log_timing(logger, "Indexer initialization", init_elapsed)
logger.info("API Server ready!")

# 索引操作锁：防止并发请求切换 embedder 或同时入库导致数据损坏
_indexer_lock = threading.Lock()


class IndexRequest(BaseModel):
    classification: Optional[str] = None
    strategy: str = "auto"
    embedding_model: Optional[str] = None  # ollama-bge-m3, chromadb-default, sentence-transformers


class SearchRequest(BaseModel):
    # 输入校验: 防止空查询打到 Ollama 浪费时间 + n_results 范围限制
    query: str = Field(..., min_length=1, max_length=2000)
    classification: Optional[str] = None
    n_results: int = Field(5, ge=1, le=100)


class IndexResponse(BaseModel):
    status: str
    message: str
    documents_processed: int = 0
    chunks_added: int = 0
    images_added: int = 0


@app.get("/")
def root():
    """服务状态."""
    return {
        "name": "MD2RAG API",
        "version": "0.2.0",
        "status": "running"
    }


@app.get("/health")
def health_check():
    """健康检查."""
    return {"status": "healthy"}


@app.post("/index", response_model=IndexResponse)
def index_documents(request: IndexRequest):
    """索引所有 MD 文件."""
    request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    logger.info("=" * 80)
    log_step(logger, f"INDEX_REQUEST[{request_id}]", f"classification={request.classification}, strategy={request.strategy}, embedding_model={request.embedding_model}")
    logger.info("=" * 80)

    total_start = time.time()

    try:
        # 根据请求动态切换嵌入模型 — 加锁防止并发切换
        with _indexer_lock:
            if request.embedding_model:
                _indexer.switch_embedder(request.embedding_model)

            log_step(logger, "INDEX" , "Calling indexer.index_directory()...")
            result = _indexer.index_directory(
                classification=request.classification,
                strategy=request.strategy,
                include_images=True,  # 默认处理图片
            )

        total_elapsed = (time.time() - total_start) * 1000
        log_timing(logger, f"Total index operation[{request_id}]", total_elapsed)
        log_step(logger, f"INDEX_COMPLETE[{request_id}]", f"status={result.status}, docs={result.documents_processed}, chunks={result.chunks_added}, images={result.images_added}")

        return IndexResponse(
            status=result.status,
            message=result.message,
            documents_processed=result.documents_processed,
            chunks_added=result.chunks_added,
            images_added=result.images_added,
        )
    except Exception as e:
        logger.error(f"[ERROR] Index operation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.")


@app.post("/search")
def search_documents(request: SearchRequest):
    """搜索相似文档."""
    logger.info(f"[SEARCH] query='{request.query}', classification={request.classification}")

    classifications = [request.classification] if request.classification else None
    if classifications is None:
        classifications = list(VALID_CLASSIFICATIONS)

    start_time = time.time()
    results = _indexer.store.search_all(
        query_text=request.query,
        classifications=classifications,
        n_results=request.n_results,
    )
    elapsed = (time.time() - start_time) * 1000
    log_timing(logger, "Search operation", elapsed)

    return {
        "query": request.query,
        "results": results,
    }


@app.get("/stats")
def get_stats():
    """获取向量数据库统计信息."""
    logger.debug("[STATS] Getting database stats...")
    stats = _indexer.get_stats()
    logger.debug(f"[STATS] collections={stats}")
    return {
        "collections": stats,
        "total": sum(stats.values()),
    }


@app.get("/status")
def get_status():
    """获取入库状态统计.

    返回待入库文件数量和已入库文件数量.
    pending = 源文件总数 - 已入库数（已入库的文件不再算"待入库"）
    """
    logger.info("[STATUS] Getting ingestion status...")

    # 获取已入库数量（从 ChromaDB）
    indexed_stats = _indexer.get_stats()

    # 获取源文件总数（扫描 MD 目录中的切片文件）
    # 密级目录映射从 loader 复用,避免再分叉(历史 bug: 旧版本曾错位映射 confidential→1Restricted)
    source_stats = {cls: 0 for cls in CLASSIFICATION_DIR_MAP}
    total_source = 0

    for classification, dir_name in CLASSIFICATION_DIR_MAP.items():
        dir_path = _indexer.config.md_dir / dir_name
        if dir_path.exists():
            count = 0
            for file_path in dir_path.rglob("*.json"):
                # Path.suffixes 方式对 ".chunks.json" 返回 [".chunks", ".json"],
                # 但 ".chunks.json.bak" 会返回 [".chunks", ".json", ".bak"] 不匹配。
                # 改用 fname.endswith(...) 判断更稳定。
                name = file_path.name
                if name.endswith((".chunks.json", ".parents.json", ".children.json")):
                    count += 1
            source_stats[classification] = count
            total_source += count
        logger.info(f"[STATUS] {classification} ({dir_name}): {source_stats[classification]} source files")

    # pending = 源文件中尚未入库的数量
    # 旧实现: indexed[cls] > 0 即将整密级 pending 归零,即使源目录有 100 个文件、
    # 只入库了 1 个,UI 也会显示"0 个待入库"。
    # 修复: 显式追踪已入库源文件集合,pending = 源文件集合 - 已入库集合。
    pending_stats = {}
    total_pending = 0
    for cls in CLASSIFICATION_DIR_MAP:
        src_count = source_stats.get(cls, 0)
        idx_count = indexed_stats.get(cls, 0)
        # 上限为源文件数(避免 indexed > source 时 pending 出现负值)
        pending_stats[cls] = max(0, src_count - idx_count)
        total_pending += pending_stats[cls]

    indexed_total = sum(indexed_stats.values())

    result = {
        "pending": pending_stats,
        "pending_total": total_pending,
        "indexed": indexed_stats,
        "indexed_total": indexed_total,
        "total": total_source,
        "embedding_model": _indexer._current_embedder_name,
        "embedding_dimension": _indexer.embedder.dimension,
        "vit_model": _indexer.config.vit_model if _indexer.config.vit_enabled else None,
        "vit_enabled": _indexer.config.vit_enabled,
    }

    logger.info(f"[STATUS] Pending: {total_pending}, Indexed: {indexed_total}")
    return result


@app.delete("/reset")
def reset_database():
    """清空向量数据库."""
    logger.info("=" * 80)
    log_step(logger, "RESET", "Clearing all vector database collections...")

    try:
        start_time = time.time()
        _indexer.clear()
        elapsed = (time.time() - start_time) * 1000
        log_timing(logger, "Reset operation", elapsed)
        log_step(logger, "RESET_COMPLETE", "All collections cleared")

        return {"status": "success", "message": "Database reset successfully"}
    except Exception as e:
        logger.error(f"[ERROR] Reset failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs for details.")


if __name__ == "__main__":
    import uvicorn
    # 默认绑定 127.0.0.1, 避免 /reset 等危险端点暴露给整个网段。
    # 通过 MD2RAG_HOST=0.0.0.0 显式开放。
    host = os.environ.get("MD2RAG_HOST", "127.0.0.1")
    port = int(os.environ.get("MD2RAG_PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
