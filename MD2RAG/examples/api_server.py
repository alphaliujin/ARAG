"""MD2RAG API 服务示例 - 使用 FastAPI 提供 REST API."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from md2rag.config import MD2RAGConfig
from md2rag.indexer import Indexer

app = FastAPI(title="MD2RAG API", version="0.2.0")

# 全局配置和索引器
_config = MD2RAGConfig()
_indexer = Indexer(_config)


class IndexRequest(BaseModel):
    classification: str | None = None
    strategy: str = "auto"


class SearchRequest(BaseModel):
    query: str
    classification: str | None = None
    n_results: int = 5


class IndexResponse(BaseModel):
    status: str
    message: str
    documents_processed: int = 0
    chunks_added: int = 0


@app.post("/index", response_model=IndexResponse)
def index_documents(request: IndexRequest):
    """索引所有 MD 文件."""
    result = _indexer.index_directory(
        classification=request.classification,
        strategy=request.strategy,
    )
    return IndexResponse(
        status=result.status,
        message=result.message,
        documents_processed=result.documents_processed,
        chunks_added=result.chunks_added,
    )


@app.post("/search")
def search_documents(request: SearchRequest):
    """搜索相似文档."""
    classifications = [request.classification] if request.classification else None
    if classifications is None:
        classifications = ["public", "confidential", "secret"]

    results = _indexer.store.search_all(
        query_text=request.query,
        classifications=classifications,
        n_results=request.n_results,
    )

    return {
        "query": request.query,
        "results": results,
    }


@app.get("/stats")
def get_stats():
    """获取向量数据库统计信息."""
    stats = _indexer.get_stats()
    return {
        "collections": stats,
        "total": sum(stats.values()),
    }
