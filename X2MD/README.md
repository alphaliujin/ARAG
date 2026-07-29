# X2MD - 文档转 Markdown 与 RAG 切片工具

X2MD 是一个多格式文档转换工具，专为 RAG（检索增强生成）场景设计，支持父子块切片策略、LLM 摘要生成、ViT 图像分析等高级功能。

## 目录

- [功能特性](#功能特性)
- [架构设计](#架构设计)
- [安装部署](#安装部署)
- [使用指南](#使用指南)
- [配置详解](#配置详解)
- [代码结构](#代码结构)
- [实现原理](#实现原理)
- [API 参考](#api参考)
- [故障排查](#故障排查)

## 功能特性

### 核心功能

| 功能模块 | 说明 | 状态 |
|---------|------|------|
| 多格式转换 | PDF/DOCX/XLSX/PPTX/HTML/图片/文本 → Markdown | ✅ 稳定 |
| 父子块切片 | 小粒度子块检索 + 大粒度父块上下文 | ✅ 稳定 |
| LLM 摘要 | 使用 Ollama 生成文档摘要注入 chunk | ✅ 稳定 |
| ViT 图像分析 | CLIP 模型提取图像特征和零样本分类 | ⚠️ 需配置镜像 |
| 文本清洗 | 全角半角转换、乱码检测、去重、标题规范化 | ✅ 稳定 |
| 表格提取 | 保留表格结构，转换为 Markdown 表格 | ✅ 稳定 |
| 页眉页脚过滤 | LLM 智能识别并移除噪声内容 | ✅ 稳定 |

### 支持的文件格式

| 格式 | 扩展名 | 处理方式 | 依赖库 |
|------|--------|---------|--------|
| PDF | `.pdf` | pdfplumber 提取文本和表格 | pdfplumber |
| Word | `.docx` | python-docx 读取段落和表格 | python-docx |
| Excel | `.xls`, `.xlsx` | openpyxl 逐行读取 | openpyxl |
| PowerPoint | `.ppt`, `.pptx` | python-pptx 逐幻灯片读取 | python-pptx |
| HTML | `.html`, `.htm` | BeautifulSoup 解析 | beautifulsoup4 |
| 图片 | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp` | Tesseract OCR | Pillow, pytesseract |
| 文本 | `.txt`, `.md` | 直接读取 | 内置 |

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI 入口层                            │
│                    x2md.cli:main()                          │
│              Click 命令行参数解析                             │
├─────────────────────────────────────────────────────────────┤
│                        配置管理层                            │
│                    x2md.config.Config                         │
│              x2md.conf 文件解析                              │
├─────────────────────────────────────────────────────────────┤
│                        文档转换层                            │
│              converters/ 目录下的各格式转换器                  │
│         PDF → DOCX → XLSX → PPTX → HTML → Image → Text       │
├─────────────────────────────────────────────────────────────┤
│                        文本处理层                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │
│  │ text_cleaner │ │     llm      │ │     vit      │          │
│  │   文本清洗    │ │  LLM 客户端   │ │  ViT 提取器   │          │
│  └──────────────┘ └──────────────┘ └──────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                        切片引擎层                            │
│                    x2md.chunk.ChunkSplitter                   │
│         SmallerChunksStrategy（父子块策略）                   │
├─────────────────────────────────────────────────────────────┤
│                        输出层                                │
│              .md + .parents.json + .children.json            │
└─────────────────────────────────────────────────────────────┘
```

### 数据流转

```
输入文档 (PDF/DOCX/...)
    ↓
[Converter] 格式转换 → Markdown 文本
    ↓
[TextCleaner] 文本清洗（可选）
    ↓
[LLM] 生成摘要（可选）
    ↓
[ChunkSplitter] 文档切片
    ↓
    ├─→ 父块（大粒度，chunk_size * 4）
    │       doc_id = UUID
    │
    └─→ 子块（小粒度，chunk_size）
            doc_id = 父块 UUID
    ↓
输出：.md + .parents.json + .children.json
```

## 安装部署

### 环境要求

- Python >= 3.9
- Ollama（可选，用于 LLM 摘要）
- Tesseract（可选，用于 OCR）

### 安装依赖

```bash
# 基础依赖
pip install click pdfplumber python-docx openpyxl python-pptx beautifulsoup4 Pillow pydantic

# OCR 支持（可选）
pip install pytesseract

# ViT 支持（可选）
pip install transformers torch

# 开发依赖
pip install pytest ruff
```

### 安装 Tesseract（macOS）

```bash
brew install tesseract tesseract-lang
# 中文 OCR 需要安装语言包
```

### 安装 Ollama

```bash
# macOS
curl -fsSL https://ollama.com/install.sh | sh

# 拉取模型
ollama pull qwen3:14b
ollama pull bge-m3  # 用于 AlphaRAG 向量化
```

## 使用指南

### 命令行参数

```bash
python -m x2md.cli [INPUT_PATH] [OPTIONS]
```

#### 位置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `INPUT_PATH` | 输入文件或目录路径 | `config.input_dir` |

#### 选项参数

| 选项 | 说明 | 示例 |
|------|------|------|
| `-o, --output` | 输出路径 | `-o ./output` |
| `--config` | 配置文件路径 | `--config ./x2md.conf` |
| `--ocr-lang` | OCR 语言 | `--ocr-lang chi_sim+eng` |
| `--encoding` | 文本编码 | `--encoding utf-8` |
| `--no-tables` | 跳过 PDF 表格提取 | `--no-tables` |
| `--sheet` | Excel 指定工作表 | `--sheet Sheet1` |
| `--no-llm` | 禁用 LLM 分析 | `--no-llm` |
| `--no-vit` | 禁用 ViT 图像分析 | `--no-vit` |
| `--vit-only` | 仅运行 ViT 分析 | `--vit-only` |
| `--list-formats` | 列出支持的格式 | `--list-formats` |
| `--chunk` | 普通切片模式 | `--chunk` |
| `--parent-child` | 父子块切片模式 | `--parent-child` |

### 使用示例

#### 基础转换

```bash
# 转换单个文件
python -m x2md.cli document.pdf

# 转换整个目录
python -m x2md.cli ./documents/

# 指定输出路径
python -m x2md.cli input.pdf -o output.md
```

#### 父子块切片模式（推荐用于 RAG）

```bash
# 启用父子块模式
python -m x2md.cli --parent-child

# 禁用 LLM，仅做切片
python -m x2md.cli --parent-child --no-llm

# 完整示例
python -m x2md.cli ../DOC --parent-child --config ./x2md.conf
```

#### 普通切片模式

```bash
python -m x2md.cli --chunk
```

#### 仅 ViT 图像分析

```bash
# 跳过文档转换，仅分析已提取的图片
python -m x2md.cli --vit-only --no-llm
```

## 配置详解

### 配置文件位置

X2MD 会自动查找 `x2md.conf` 配置文件：
1. 当前目录
2. 上级目录（递归向上查找）
3. 使用默认配置

### 完整配置示例

```ini
[x2md]
input_dir = ../DOC              # 输入目录
output_dir = ../MD              # 输出目录
encoding = utf-8                # 默认编码

[llm]
enabled = true                  # 是否启用 LLM
model = qwen3:14b             # Ollama 模型名称
base_url = http://localhost:11434  # Ollama 服务地址
timeout = 120                   # 请求超时（秒）
call_interval = 0.5             # 调用间隔（秒）
max_concurrent = 10             # 最大并发数

[pdf]
extract_tables = true           # 是否提取表格
page_separator = \n\n---\n\n   # 分页分隔符

[docx]
heading_style = true            # 是否识别标题样式

[xlsx]
include_all_sheets = true       # 是否包含所有工作表
default_sheet =                 # 默认工作表名称

[pptx]
slide_separator = \n\n---\n\n   # 幻灯片分隔符

[image]
ocr_enabled = true              # 是否启用 OCR
ocr_lang = eng                  # OCR 语言

[vit]
enabled = true                  # 是否启用 ViT
model = openai/clip-vit-base-patch32  # CLIP 模型名称
device = cpu                    # 运行设备：cpu/cuda/mps
local_files_only = false        # 是否仅使用本地模型
mirror = https://hf-mirror.com  # HuggingFace 镜像地址
labels =                        # 自定义分类标签（逗号分隔）

[html]
encoding = utf-8                # HTML 文件编码

[chunk]
chunk_size = 500                # 切片大小（字符）
chunk_overlap = 100             # 重叠大小（字符）
separators = \n\n,\n,。,., ,   # 分隔符列表（逗号分隔）
separator_rule = after,after,after,after,after,after  # 分隔规则
max_chunk_limit = 10000         # 最大 chunk 长度限制
inject_abstract = true          # 是否注入 LLM 摘要
```

### 配置项详细说明

#### [x2md] 基础配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `input_dir` | str | `../from` | 输入目录，支持相对路径 |
| `output_dir` | str | `../to` | 输出目录，支持相对路径 |
| `encoding` | str | `utf-8` | 默认文本编码 |

#### [llm] LLM 配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enabled` | bool | `true` | 是否启用 LLM |
| `model` | str | `qwen3:14b` | Ollama 模型名称 |
| `base_url` | str | `http://localhost:11434` | Ollama API 地址 |
| `timeout` | int | `120` | 请求超时时间（秒） |
| `call_interval` | float | `0.5` | 连续调用间隔（秒） |
| `max_concurrent` | int | `10` | 最大并发请求数 |

**timeout 说明**：
- 生成摘要时，模型需要加载和处理，首次调用可能较慢
- 120 秒对于简单文档足够，复杂文档可能需要 300 秒或更长
- 超时后会静默失败，不影响其他功能

#### [chunk] 切片配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `chunk_size` | int | `500` | 子块目标大小（字符） |
| `chunk_overlap` | int | `100` | 相邻块重叠字符数 |
| `separators` | list | `\n\n,\n,。,., ,` | 分隔符优先级列表 |
| `separator_rule` | list | `after,after,...` | 每个分隔符的规则：`after`/`before` |
| `max_chunk_limit` | int | `10000` | 单个 chunk 最大长度限制 |
| `inject_abstract` | bool | `true` | 是否注入 LLM 生成的摘要 |

**父子块大小计算**：
- 父块大小 = `chunk_size * 4` = 2000 字符
- 子块大小 = `chunk_size` = 500 字符

## 代码结构

### 目录结构

```
X2MD/
├── src/x2md/                 # 主代码目录
│   ├── __init__.py           # 包入口，导出公共 API
│   ├── cli.py                # 命令行接口
│   ├── config.py             # 配置管理
│   ├── chunk.py              # 切片引擎
│   ├── llm.py                # LLM 客户端
│   ├── vit.py                # ViT 图像特征提取
│   ├── text_cleaner.py       # 文本清洗
│   ├── utils.py              # 工具函数
│   └── converters/           # 格式转换器
│       ├── __init__.py       # 转换器注册
│       ├── base.py           # 基类
│       ├── pdf.py            # PDF 转换
│       ├── docx.py           # Word 转换
│       ├── xlsx.py           # Excel 转换
│       ├── pptx.py           # PPT 转换
│       ├── html.py           # HTML 转换
│       ├── image.py          # 图片 OCR
│       └── text.py           # 纯文本
├── tests/                    # 测试目录
├── pyproject.toml            # 项目配置
├── x2md.conf                 # 运行时配置
└── README.md                 # 本文档
```

### 核心模块说明

#### cli.py - 命令行接口

```python
@click.command()
@click.argument("input_path", type=click.Path(), required=False)
@click.option("-o", "--output", ...)
@click.option("--parent-child", is_flag=True, ...)
def main(input_path, output, ..., parent_child):
    """主入口函数"""
    # 1. 加载配置
    # 2. 文档转换
    # 3. 切片处理
    # 4. ViT 分析（可选）
```

**核心流程**：
1. 解析命令行参数
2. 加载配置文件
3. 初始化 LLM 客户端（如果启用）
4. 遍历输入目录，转换文档
5. 对 Markdown 进行切片（普通或父子块模式）
6. 运行 ViT 图像分析（如果启用）

#### config.py - 配置管理

```python
class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str | Path | None = None):
        # 加载默认配置
        # 读取配置文件（如果存在）
    
    @property
    def input_dir(self) -> Path:
        # 解析输入目录路径
    
    @property
    def llm_enabled(self) -> bool:
        # LLM 是否启用
    
    def create_llm_client(self) -> OllamaClient | None:
        # 创建 LLM 客户端
```

**配置加载优先级**：
1. 代码默认值（`_DEFAULTS` 字典）
2. 配置文件（`x2md.conf`）
3. 命令行参数

#### chunk.py - 切片引擎

```python
class Chunk(BaseModel):
    """切片数据模型"""
    text: str
    metadata: ChunkMetadata

class ChunkSplitter:
    """切片引擎"""
    
    def __init__(self, chunk_size=500, chunk_overlap=100, ...):
        # 初始化切片参数
    
    def split_text(self, text: str) -> list[str]:
        # 递归分隔符切片
    
    def split_documents(self, text, source, page, ...) -> list[Chunk]:
        # 生成带元数据的 Chunk 列表

class SmallerChunksStrategy:
    """父子块策略"""
    
    def __init__(self, parent_chunk_size=2000, child_chunk_size=500, ...):
        self.parent_splitter = ChunkSplitter(parent_chunk_size, ...)
        self.child_splitter = ChunkSplitter(child_chunk_size, ...)
    
    def split(self, text, source, ...) -> tuple[list[Chunk], list[Chunk]]:
        # 返回 (parent_chunks, child_chunks)
```

**切片算法**：
1. 按分隔符优先级递归切分
2. 合并短片段，拆分长片段
3. 添加重叠内容
4. 映射 bbox 和页码（PDF）

#### llm.py - LLM 客户端

```python
class OllamaClient:
    """Ollama API 客户端"""
    
    def __init__(self, model, base_url, timeout=120, ...):
        # 初始化客户端参数
    
    def generate(self, prompt: str, system: str = "", use_cache: bool = True) -> str:
        # 调用 /api/generate 生成文本
    
    def is_available(self) -> bool:
        # 检查 Ollama 服务是否可用
    
    def _cache_key(self, prompt: str) -> str:
        # 生成缓存键（SHA256）
```

**API 调用**：
- `/api/generate` - 文本生成（摘要、噪声检测）
- `/api/tags` - 检查服务状态
- 支持请求缓存，避免重复调用

#### converters/ - 格式转换器

```python
class BaseConverter(ABC):
    """转换器基类"""
    extensions: ClassVar[list[str]] = []
    
    @abstractmethod
    def convert(self, file_path: Path, **kwargs) -> str:
        ...
    
    @classmethod
    def supports(cls, file_path: Path) -> bool:
        return file_path.suffix.lower() in cls.extensions
```

**转换器注册机制**：
```python
_CONVERTERS: list[BaseConverter] = [
    PdfConverter(),
    DocxConverter(),
    XlsxConverter(),
    PptxConverter(),
    ImageConverter(),
    HtmlConverter(),
    TextConverter(),
]

_EXTENSION_MAP: dict[str, BaseConverter] = {}
for _conv in _CONVERTERS:
    for _ext in _conv.extensions:
        _EXTENSION_MAP[_ext] = _conv
```

## 实现原理

### 父子块策略详解

#### 设计动机

在 RAG 系统中，切片大小是一个权衡：
- **小切片**：检索精度高，但上下文不完整
- **大切片**：上下文完整，但检索精度低

**父子块策略**同时获得两者的优势：
- **子块**：小粒度（500字符），用于语义检索
- **父块**：大粒度（2000字符），用于提供完整上下文

#### 实现机制

```python
class SmallerChunksStrategy:
    def split(self, text, source, abstract, document_name):
        # 1. 生成父块
        parent_chunks = self.parent_splitter.split_documents(
            text=text,  # 完整文档
            ...
        )
        
        # 2. 为每个父块分配 UUID
        for parent in parent_chunks:
            parent.metadata.doc_id = str(uuid.uuid4())
        
        # 3. 对每个父块生成子块
        for parent in parent_chunks:
            parent_children = self.child_splitter.split_documents(
                text=parent.text,  # 父块内容
                ...
            )
            
            # 4. 子块继承父块的 doc_id
            for child in parent_children:
                child.metadata.doc_id = parent.metadata.doc_id
        
        return parent_chunks, child_chunks
```

#### 关联关系

```
Parent Chunk (doc_id: uuid-abc)
    ├── Child Chunk 1 (doc_id: uuid-abc)
    ├── Child Chunk 2 (doc_id: uuid-abc)
    └── Child Chunk 3 (doc_id: uuid-abc)

Parent Chunk (doc_id: uuid-def)
    ├── Child Chunk 4 (doc_id: uuid-def)
    └── Child Chunk 5 (doc_id: uuid-def)
```

**检索流程**：
1. 使用子块进行语义检索（向量相似度）
2. 找到匹配的子块
3. 通过 `doc_id` 查找对应的父块
4. 返回父块内容作为上下文

### 切片算法详解

#### 递归分隔符切片

```python
def _split_text_recursive(self, text: str, separators: list[str]) -> list[str]:
    if len(text) <= self.chunk_size:
        return [text]
    
    # 按优先级尝试每个分隔符
    for i, sep in enumerate(separators):
        splits = re.split(sep, text)
        
        if len(splits) > 1:
            # 使用更细粒度的分隔符递归处理
            new_seps = separators[i + 1:]
            result = []
            
            for split in splits:
                if len(split) > self.chunk_size:
                    # 过长则递归切分
                    result.extend(self._split_text_recursive(split, new_seps))
                else:
                    result.append(split)
            
            return result
    
    # 无分隔符可切，按字符硬切
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
```

#### 分隔符规则

```python
# 默认分隔符（按优先级）
separators = ["\n\n", "\n", "。", ".", " ", ""]

# 规则：after = 分隔符附在前一段末尾
#       before = 分隔符附在后一段开头
separator_rule = ["after", "after", "after", "after", "after", "after"]
```

示例：
```
文本："第一段\n\n第二段\n\n第三段"

after 规则：
- chunk1: "第一段\n\n"
- chunk2: "第二段\n\n"
- chunk3: "第三段"
```

#### 重叠处理

```python
def _add_overlap(self, chunks: list[str]) -> list[str]:
    overlapped = []
    for i, chunk in enumerate(chunks):
        prefix = chunks[i-1][-overlap:] if i > 0 else ""  # 前一块尾部
        suffix = chunks[i+1][:overlap] if i < len(chunks)-1 else ""  # 后一块头部
        overlapped.append(prefix + chunk + suffix)
    return overlapped
```

### LLM 摘要生成

#### 提示词设计

```python
abstract_prompt = (
    "请用一句话概括以下文档的核心内容：\n\n"
    + content[:3000]  # 截取前 3000 字符
)
```

#### 缓存机制

```python
def generate(self, prompt: str, use_cache: bool = True) -> str:
    cache_key = self._cache_key(prompt)  # SHA256 哈希
    
    if use_cache and cache_key in self._cache:
        return self._cache[cache_key]  # 命中缓存
    
    # 调用 API
    response = self._post("/api/generate", payload)
    
    if use_cache:
        self._cache[cache_key] = response
    
    return response
```

### PDF 元素级追踪

```python
def convert_with_elements(self, file_path: Path, **kwargs) -> dict:
    """转换 PDF 并返回 element 级元数据"""
    
    element_indexes: list[list[int]] = []  # 字符范围 [[start, end], ...]
    element_pages: list[int] = []           # 页码 [1, 1, 2, 2, ...]
    element_bboxes: list[list[float]] = []  # 坐标 [[x1,y1,x2,y2], ...]
    element_types: list[str] = []           # 类型 ["text", "Table", ...]
    
    char_pos = 0
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # 提取文本
            text = page.extract_text()
            for line in text.split("\n"):
                element_indexes.append([char_pos, char_pos + len(line)])
                element_pages.append(page_num)
                element_bboxes.append(self._get_line_bbox(page, line))
                element_types.append("text")
                char_pos += len(line) + 1
            
            # 提取表格
            for table in page.extract_tables():
                md_table = table_to_lines(table)
                element_indexes.append([char_pos, char_pos + len(md_table)])
                element_pages.append(page_num)
                element_bboxes.append(self._get_table_bbox(page, table))
                element_types.append("Table")
                char_pos += len(md_table) + 1
```

## API 参考

### 公共 API

```python
from x2md import (
    convert,              # 转换文件
    get_converter,        # 获取转换器
    supported_extensions, # 支持的扩展名列表
    Config,               # 配置类
    load_config,          # 加载配置
    ChunkSplitter,        # 切片引擎
    SmallerChunksStrategy, # 父子块策略
    TextCleaner,          # 文本清洗
    OllamaClient,         # LLM 客户端
    VitExtractor,         # ViT 提取器
)
```

### convert 函数

```python
def convert(file_path: str | Path, **kwargs) -> str:
    """转换文件为 Markdown
    
    Args:
        file_path: 文件路径
        **kwargs: 传递给转换器的参数
            - llm_client: OllamaClient 实例
            - extract_tables: 是否提取表格
            - ocr_lang: OCR 语言
            - encoding: 文本编码
    
    Returns:
        Markdown 文本
    
    Raises:
        ValueError: 不支持的文件格式
    """
```

### ChunkSplitter

```python
class ChunkSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        separator_rule: list[str] | None = None,
        max_chunk_limit: int = 10000,
    ):
        ...
    
    def split_text(self, text: str) -> list[str]:
        """纯文本切片"""
    
    def split_documents(
        self,
        text: str,
        source: str = "",
        page: int | None = None,
        abstract: str = "",
        document_name: str = "",
        element_indexes: list[list[int]] | None = None,
        element_pages: list[int] | None = None,
        element_bboxes: list[list[float]] | None = None,
        element_types: list[str] | None = None,
    ) -> list[Chunk]:
        """生成带元数据的切片"""
```

### SmallerChunksStrategy

```python
class SmallerChunksStrategy:
    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500,
        child_overlap: int = 50,
        parent_overlap: int = 100,
        separators: list[str] | None = None,
        separator_rule: list[str] | None = None,
    ):
        ...
    
    def split(
        self,
        text: str,
        source: str = "",
        page: int | None = None,
        abstract: str = "",
        document_name: str = "",
    ) -> tuple[list[Chunk], list[Chunk]]:
        """返回 (parent_chunks, child_chunks)"""
```

## 故障排查

### 常见问题

#### 1. LLM 连接失败

**现象**：
```
Warning: LLM enabled but Ollama not available at http://localhost:11434
```

**解决**：
```bash
# 检查 Ollama 服务
ollama serve

# 检查模型是否存在
ollama list

# 拉取模型
ollama pull qwen3:14b
```

#### 2. LLM 生成超时

**现象**：摘要为空，日志显示超时

**解决**：
- 增加 `timeout` 配置
- 使用更轻量的模型
- 禁用摘要：`inject_abstract = false`

#### 3. ViT 模型下载失败

**现象**：
```
HTTPSConnectionPool(host='huggingface.co', ...)
```

**解决**：
```ini
[vit]
mirror = https://hf-mirror.com
```

或下载模型到本地：
```ini
model = /path/to/local/clip-vit-base-patch32
local_files_only = true
```

#### 4. 中文乱码

**解决**：
```ini
[x2md]
encoding = utf-8

[pdf]
# 确保 pdfplumber 正确识别编码
```

#### 5. 表格格式错乱

**原因**：单元格内容包含 `|` 字符

**解决**：自动转义（已实现）

### 调试模式

```bash
# 查看详细日志
python -m x2md.cli --parent-child -v

# 测试单个文件
python -m x2md.cli test.pdf --no-llm --no-vit
```

## 更新日志

### v0.2.0
- 基础文档转换功能
- 父子块切片策略
- LLM 摘要生成
- ViT 图像分析

## License

MIT License
