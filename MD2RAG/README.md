# MD2RAG

将 X2MD 生成的 Markdown 切片向量化并存入向量数据库。

## 功能特性

- **适配 X2MD 多种切片模式**：
  - 普通切片 (`.chunks.json`)
  - 父子块切片 (`.parents.json` + `.children.json`)
- **多种嵌入模型支持**：
  - ChromaDB 默认模型 (all-MiniLM-L6-v2)
  - Ollama 嵌入
  - Sentence-Transformers
- **密级分类管理**：
  - 自动识别 MD 目录密级结构（0Public/1Secret/2Confidential）
  - 按密级分别存储到不同集合
- **CLI 工具**：提供命令行工具进行索引、搜索、统计

## 安装

```bash
cd MD2RAG
pip install -e .
```

可选依赖：
```bash
# Ollama 嵌入
pip install -e ".[ollama]"

# Sentence-Transformers
pip install -e ".[st]"
```

## 快速开始

### Python API

```python
from md2rag import Indexer, MD2RAGConfig

# 配置
config = MD2RAGConfig()
config.md_dir = "../MD"
config.vector_db_dir = "./vector_db"

# 创建索引器
indexer = Indexer(config)

# 索引所有文件
result = indexer.index_directory()
print(f"索引了 {result.chunks_added} 个切片")

# 搜索
results = indexer.store.search_all("查询文本", n_results=5)
```

### CLI 工具

```bash
# 查看将要索引的文件（试运行）
md2rag --config md2rag.conf dry-run

# 索引所有文件
md2rag --config md2rag.conf index-all

# 按密级索引
md2rag --config md2rag.conf index-all -c secret

# 查看统计
md2rag --config md2rag.conf stats

# 清空向量数据库
md2rag --config md2rag.conf clear
```

## 目录结构

```
MD2RAG/
├── md2rag/
│   ├── __init__.py       # 模块入口
│   ├── config.py         # 配置管理
│   ├── chunk_loader.py   # 切片文件加载（适配 X2MD 输出）
│   ├── embedder.py       # 嵌入模型
│   ├── vector_store.py   # 向量数据库存储
│   ├── indexer.py        # 索引器主逻辑
│   └── cli.py            # 命令行工具
├── examples/
│   ├── basic_usage.py    # 基础使用示例
│   └── api_server.py     # FastAPI 服务示例
├── pyproject.toml
└── README.md
```

## 配置

创建 `md2rag.conf` 配置文件：

```ini
[md2rag]
md_dir = ../MD
vector_db_dir = ./vector_db
collection_prefix = md2rag
default_chunk_strategy = auto

[embedding]
model = chromadb-default
batch_size = 32

[ollama]
enabled = false
base_url = http://localhost:11434
model = qwen2.5:7b

[sentence_transformers]
enabled = false
model_name = all-MiniLM-L6-v2
```

## 与 X2MD 的集成

X2MD 生成的切片文件格式：

### 普通切片 (chunk)
```
MD/0Public/doc.md
MD/0Public/doc.chunks.json
```

### 父子块切片 (parent-child)
```
MD/1Secret/doc.md
MD/1Secret/doc.parents.json
MD/1Secret/doc.children.json
```

MD2RAG 会自动检测这些文件并进行索引。

## 与 backend 的集成

MD2RAG 使用与 backend 相同的 ChromaDB 存储格式，可以直接复用 backend 的向量数据库：

```python
from md2rag import VectorStore

store = VectorStore(
    db_dir="../backend/vector_db",
    collection_prefix="md2rag",
)
```
