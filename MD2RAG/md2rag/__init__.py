"""MD2RAG - 将 X2MD 生成的 Markdown 父子块切片向量化并存入向量数据库.

借鉴 bisheng 项目的入库方式：
  - 父子块双 collection (parent + child)
  - 通过 doc_id 关联
  - bbox 元数据追踪（行级合成坐标，可升级为真实 PDF bbox）
  - LLM 摘要（可选，使用 Ollama 本地模型）
  - ViT-Large 图片入库

典型用法:
    >>> from md2rag.config import load_config
    >>> from md2rag.indexer import Indexer
    >>>
    >>> config = load_config()
    >>> indexer = Indexer(config)
    >>> result = indexer.index_directory(classification="public")
    >>> print(result.message)
"""

from md2rag.bbox_extractor import (
    LineInfo,
    assign_bboxes_to_chunk,
    get_bboxes_for_record,
    make_synthetic_bboxes,
)
from md2rag.config import MD2RAGConfig, load_config
from md2rag.embedder import (
    ChromaDefaultEmbedder,
    Embedder,
    MPSEmbedder,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)
from md2rag.image_processor import ImageProcessor, ViTImageEmbedder
from md2rag.indexer import Indexer, IndexResult
from md2rag.interval_search import IntervalSearch
from md2rag.loader import ChunkFileType, ChunkLoader, ChunkRecord
from md2rag.llm_summary import (
    LLMClient,
    OllamaChatClient,
    async_extract_title,
    batch_extract_abstracts,
    extract_abstract,
    extract_title,
)
from md2rag.logger import get_logger, log_memory, log_step, log_timing
from md2rag.parent_child_retriever import ParentChildRetriever, RetrievalHit
from md2rag.text_splitter import Chunk, Document, ElemCharacterTextSplitter
from md2rag.vector_store import VectorStore

__version__ = "0.2.0"
__all__ = [
    # 配置
    "MD2RAGConfig",
    "load_config",
    # 数据结构
    "ChunkRecord",
    "Chunk",
    "Document",
    "IndexResult",
    "RetrievalHit",
    "LineInfo",
    "ChunkFileType",
    # 核心类
    "ChunkLoader",
    "Indexer",
    "ParentChildRetriever",
    "VectorStore",
    # 嵌入
    "Embedder",
    "ChromaDefaultEmbedder",
    "OllamaEmbedder",
    "SentenceTransformerEmbedder",
    "MPSEmbedder",
    "create_embedder",
    # 文本处理
    "ElemCharacterTextSplitter",
    "IntervalSearch",
    "make_synthetic_bboxes",
    "assign_bboxes_to_chunk",
    "get_bboxes_for_record",
    # 图片
    "ViTImageEmbedder",
    "ImageProcessor",
    # LLM 摘要
    "LLMClient",
    "OllamaChatClient",
    "extract_title",
    "async_extract_title",
    "extract_abstract",
    "batch_extract_abstracts",
    # 日志
    "get_logger",
    "log_step",
    "log_timing",
    "log_memory",
]
