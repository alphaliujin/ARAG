"""MD2RAG 使用示例脚本."""

from md2rag.config import MD2RAGConfig
from md2rag.indexer import Indexer


def main():
    # 方式1: 使用默认配置
    config = MD2RAGConfig()
    config.md_dir = "/Users/alpha/workspace/ARAG_V0.2/MD"
    config.vector_db_dir = "/Users/alpha/workspace/ARAG_V0.2/MD2RAG/vector_db"

    indexer = Indexer(config)

    # 索引所有文件
    print("开始索引所有 MD 切片文件...")
    result = indexer.index_directory()
    print(f"状态: {result.status}")
    print(f"消息: {result.message}")
    print(f"文档数: {result.documents_processed}")
    print(f"切片数: {result.chunks_added}")

    # 查看统计
    stats = indexer.get_stats()
    print("\n向量数据库统计:")
    for cls, count in stats.items():
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
