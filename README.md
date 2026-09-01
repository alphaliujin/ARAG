# ARAG — 敏感信息识别系统 / Sensitive Information Detection System

[简体中文](#简体中文) ｜ [English](#english)

---

# 简体中文

**敏感信息识别系统（Sensitive Information Detection System，基于 RAG 的文档涉密信息检测）**

ARAG 先把含敏感信息的文档预处理、切片、向量化，建立按密级分类的**敏感信息 RAG 库**；再对用户上传的待检文档进行层级比对，判断其与各密级知识库的相似度，从而识别其中是否含有敏感/涉密内容。

- **版本**：0.2.0
- **License**：MIT

## ✨ 功能特性

| 模块 | 说明 |
|------|------|
| 📄 多格式文档预处理 | PDF / DOCX / XLSX / PPTX / HTML / 图片 / 纯文本 → Markdown（X2MD） |
| 🧩 父子块切片 | 子块（小粒度）做语义检索 + 父块（大粒度）提供上下文，兼顾精度与完整性 |
| 🧠 向量化入库 | 通过 Ollama `bge-m3`（1024 维）嵌入，写入 ChromaDB，按密级分集合 |
| 🔐 密级分类管理 | 公开 `0Public` / 受限 `1Restricted` / 机密 `2Confidential`，分别入不同向量集合 |
| 🔄 跨密级去重 | 跨密级比对子块相似度，自动删除高密级库中与低密级重复的内容 |
| 🔍 文档比对（DocScan） | 上传待检文档 → 预处理 → 嵌入 → 摘要/父块/子块逐级比对，输出各密级匹配结果 |
| ✍️ 文本扫描 | 直接粘贴文本，即时检测敏感信息 |
| 🗂️ 后台任务 | 预处理 / 入库 / 去重均支持异步执行、进度查询、可取消 |
| ⚙️ 运行时配置 | 相似度阈值、切片参数、上传限制等可在前端「设置」页动态调整 |
| 🚀 一键部署 | 单进程 uvicorn 同时服务前端 + API，自包含，无需 nginx / Node |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        浏览器 / Browser                          │
│              http://localhost:3000 (dev) 或 :8000 (部署)         │
└───────────────────────────────┬─────────────────────────────────┘
                                │  HTTP /api/v1
┌───────────────────────────────▼─────────────────────────────────┐
│                    FastAPI 后端 (backend/, :8000)                │
│  扫描 / 预处理 / 入库 / 去重 / DocScan / 设置 / 后台任务            │
└──┬──────────────┬──────────────┬──────────────┬─────────────────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
┌─────────┐ ┌──────────┐ ┌─────────────┐ ┌──────────────┐
│  X2MD   │ │  MD2RAG  │ │   ChromaDB  │ │   Ollama     │
│ 文档转MD │ │ 切片向量化 │ │  vector_db/  │ │ bge-m3 嵌入   │
│ +父子切片│ │  + 入库   │ │  按密级分集合 │ │  + 可选 rerank │
└────┬────┘ └────┬─────┘ └─────────────┘ └──────────────┘
     │           │
     ▼           ▼
  DOC/<密级>/   MD/<密级>/
  源文档输入    .md + .parents/.children.json
```

**核心数据流**：

```
源文档 (DOC/<密级>/)
   │  ① 预处理 (X2MD: 转换 → 清洗 → 父子块切片)
   ▼
Markdown 切片 (MD/<密级>/)
   │  ② 入库 (MD2RAG: Ollama bge-m3 嵌入 → ChromaDB)
   ▼
敏感信息 RAG 库 (vector_db/, 6 个集合: 各密级 parent + child)
   │  ③ 数据优化 (跨密级去重, ≥阈值自动删除高密级重复块)
   ▼
   ┌─────────────────────────────────────┐
   │  ④ 待检文档 → 预处理 → 嵌入 →        │
   │     层级比对 (摘要→父块→子块)         │ → 各密级相似度匹配结果
   └─────────────────────────────────────┘
```

## 🧰 技术栈

- **后端**：Python ≥ 3.9、FastAPI、uvicorn、ChromaDB、sentence-transformers、pdfplumber、python-docx、openpyxl、python-pptx、beautifulsoup4
- **前端**：React 18、Ant Design 5、Axios
- **嵌入 / LLM**：Ollama（`bge-m3:latest` 嵌入、可选 `qwen3:14b` 摘要）
- **向量库**：ChromaDB（持久化于 `vector_db/`）
- **可选**：BAAI/bge-reranker-v2-m3 重排序器、Tesseract OCR

## 📁 项目结构

```
ARAG/
├── backend/                # FastAPI 后端
│   └── app/
│       ├── main.py         # 入口（lifespan 初始化向量库）
│       ├── api/endpoints.py# API 路由
│       ├── core/config.py  # 配置
│       ├── middleware/auth # API Key 认证中间件
│       └── services/       # ingestion / docscan / dedup / preprocess / scanner / reranker ...
├── X2MD/                   # 文档转 Markdown + 父子块切片工具
├── MD2RAG/                 # 切片向量化 + 入库工具
├── UI/                     # React 前端
├── DOC/                    # 源文档输入（按密级分目录，运行数据）
│   ├── 0Public/ 1Restricted/ 2Confidential/
├── MD/                     # X2MD 输出的 Markdown + 切片 JSON（运行数据）
├── vector_db/              # ChromaDB 向量库（按密级分集合，运行数据）
├── DocScan/                # 待检文档暂存 + 中间产物（运行数据）
├── dedup_results/          # 跨密级去重结果 .md / .json（运行数据）
├── bge-m3-local/           # 本地 bge-m3 嵌入模型（可选）
├── clip-vit-large-patch14-local/  # 本地 CLIP ViT 图像模型（可选）
├── runtime_settings.json   # 运行时设置（前端「设置」页可改）
├── start.sh / stop.sh      # 一键启动 / 停止脚本（macOS 本地开发）
├── scripts/                # 部署 & 诊断脚本
│   ├── make_deploy_package.sh  # 构建 Linux 部署包（ARAG_V0.2-linux.tar.gz）
│   ├── deploy-dgx.sh · dgx-status.sh  # DGX 远程更新 / 状态检查
│   └── diagnose_ingestion.py 等  # 向量库 / 索引诊断
├── deploy/                 # Linux 自包含部署包（生产部署唯一真相）
└── DEPENDENCIES.md         # 依赖清单
```

## 🚀 快速开始

### 1. 环境要求

- Python ≥ 3.9
- Node.js ≥ 16
- [Ollama](https://ollama.com)（嵌入必需）
- 可选：`antiword` / `libreoffice`（仅 `.doc` / `.ppt` 转换需要）

### 2. 安装依赖

```bash
# 拉取嵌入模型
ollama pull bge-m3:latest

# macOS 可选系统依赖
brew install antiword libreoffice tesseract tesseract-lang

# 后端（X2MD + MD2RAG + backend，均以 editable 模式安装到 backend/.venv）
cd backend && python3 -m venv .venv && source .venv/bin/activate
pip install -e ../X2MD && pip install -e ../MD2RAG && pip install -e .

# 前端
cd ../UI && npm install
```

### 3. 一键启动（开发模式）

```bash
./start.sh            # 启动后端(:8000) + 前端(:3000)
./start.sh status     # 查看服务状态
./start.sh stop        # 停止所有服务
# 也可分别启动:
./start.sh back       # 仅后端
./start.sh front      # 仅前端
```

启动后访问：
- 前端界面：http://localhost:3000
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## 📖 使用流程

1. **放置源文档**：将含敏感信息的文档放入 `DOC/<密级>/`（`0Public` / `1Restricted` / `2Confidential`）。
2. **预处理**：在前端触发预处理（X2MD 转换 + 父子块切片），输出到 `MD/<密级>/`。
3. **入库**：向量化（Ollama bge-m3）写入 ChromaDB，建立敏感信息 RAG 库。
4. **数据优化（可选）**：跨密级去重，删除高密级库中与低密级重复的子块。
5. **文档比对**：在「文件扫描」上传待检文档 → 预处理 → 生成向量 → 比对，查看各密级相似度匹配；或在「文本扫描」直接粘贴文本检测。

## 🔌 API 概览

所有接口前缀 `/api/v1`，交互文档见 `/docs`。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/scan/file` | 上传待检文件到 DocScan |
| POST | `/scan/text` | 文本敏感信息扫描 |
| POST | `/docscan/preprocess` · `/embed` · `/compare` | DocScan 预处理 / 嵌入 / 层级比对 |
| GET | `/docscan/files` · `/status` · `/stats` | DocScan 文件列表 / 状态 / 统计 |
| POST | `/preprocess` · `/preprocess/retry` | 预处理 / 失败重试 |
| POST | `/tasks/preprocess` · `/tasks/ingest` · `/tasks/dedup` | 异步后台任务（可查进度 / 可取消） |
| POST | `/ingest` · DELETE `/reset` | 入库 / 重置向量库 |
| POST | `/dedup` · `/dedup/apply` · GET `/dedup/results` | 去重比对 / 应用去重 / 查看结果 |
| GET · DELETE | `/dedup/results/file` | 查看 / 删除单个去重结果 |
| GET | `/scan-source-dir` | 扫描源目录（DOC/密级）文件统计 |
| GET | `/stats` · `/status` | 向量库统计 / 入库状态 |
| GET · PUT · POST | `/settings` · `/settings/{category}/reset` | 读取 / 更新 / 重置 运行时配置 |
| GET | `/tasks/active` · `/tasks/latest/{type}` · `/tasks/{id}` · POST `/tasks/{id}/cancel` | 任务查询与取消 |

## ⚙️ 配置

- **`backend/.env`**：`API_KEY`（认证）、`AUTH_DISABLED`（开发免认证）。
- **`runtime_settings.json`**：相似度阈值（当前默认机密 0.85 / 受限 0.7）、切片参数、上传大小、嵌入模型等，前端「设置」页可改。
- **`X2MD/x2md.conf`**：文档转换 / 切片 / LLM / ViT 配置。
- **`MD2RAG/md2rag.conf`**：嵌入模型、向量库路径、设备（`[device] type=cpu` 强制走 Ollama）。

**相似度三档**（去重/比对）：`≥0.8` 高度相似 / `0.65–0.8` 中度相似 / `0.5–0.65` 弱相关。

## 📦 部署（Linux / DGX Spark）

仓库提供自包含部署包（`ARAG_V0.2-linux.tar.gz` + `deploy/`）：

```bash
tar xzf ARAG_V0.2-linux.tar.gz
cd ARAG_V0.2
./setup.sh            # 自动装系统依赖 / venv / Python 依赖 / ollama pull bge-m3
./start.sh            # 单进程 uvicorn 服务 UI(:/) + API(/api/v1)，默认 0.0.0.0:8000
ARAG_HOST=0.0.0.0 ARAG_PORT=9000 ./start.sh   # 自定义端口
```

**已有安装的远程更新（DGX Spark 等）**：

```bash
# 本机打包
npm --prefix UI run build                          # 构建前端（相对 /api/v1，同源）
./scripts/make_deploy_package.sh                   # 产出 ARAG_V0.2-linux.tar.gz
scp ARAG_V0.2-linux.tar.gz scripts/deploy-dgx.sh scripts/dgx-status.sh alpha@<DGX-IP>:~/
# 在 DGX 上执行（密码交互）
bash deploy-dgx.sh    # 原子切换：保留数据/设置/venv，失败自动回滚
bash dgx-status.sh    # 检查 stage/bak/venv/构建产物/端口/进程/日志
```

详见 [`deploy/README.md`](deploy/README.md)。

## 🔒 安全提醒

- 开发 / 部署包默认免认证（`deploy/setup.sh` 会写入 `AUTH_DISABLED=True`；`app/core/config.py` 默认 `False`，需显式开启），仅为方便内网部署，**生产环境务必设置 `API_KEY` 并启用认证**。
- 系统处理的可能为机密文档，`DOC/`、`MD/`、`vector_db/`、`DocScan/`、`dedup_results/` 含明文内容，请注意磁盘访问控制。
- 所有文件操作接口均做了路径穿越防御（basename 校验 + `resolve` 后归属校验）。

---

# English

**Sensitive Information Detection System** — an RAG-based system that detects sensitive/classified content in documents.

ARAG first preprocesses, chunks, and vectorizes sensitive documents to build a **classification-aware sensitive-information RAG library**; it then hierarchically compares user-uploaded documents against this library to determine similarity to each classification level, thereby identifying sensitive/classified content.

- **Version**: 0.2.0
- **License**: MIT

## ✨ Features

| Module | Description |
|--------|-------------|
| 📄 Multi-format preprocessing | PDF / DOCX / XLSX / PPTX / HTML / image / text → Markdown (X2MD) |
| 🧩 Parent-child chunking | Small child chunks for precise retrieval + large parent chunks for context |
| 🧠 Embedding & ingestion | Ollama `bge-m3` (1024-dim) → ChromaDB, collections split by classification level |
| 🔐 Classification management | `0Public` / `1Restricted` / `2Confidential`, stored in separate vector collections |
| 🔄 Cross-level dedup | Compare child-chunk similarity across levels; auto-delete duplicates from higher-level libraries |
| 🔍 Document scan (DocScan) | Upload → preprocess → embed → hierarchical comparison (summary → parent → child) |
| ✍️ Text scan | Paste text directly to detect sensitive information |
| 🗂️ Background tasks | Preprocess / ingest / dedup run async with progress & cancellation |
| ⚙️ Runtime config | Similarity thresholds, chunk params, upload limits adjustable from the UI Settings page |
| 🚀 One-command deploy | Single uvicorn process serves UI + API, self-contained, no nginx / Node |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser                                   │
│              http://localhost:3000 (dev) or :8000 (deploy)        │
└───────────────────────────────┬─────────────────────────────────┘
                                │  HTTP /api/v1
┌───────────────────────────────▼─────────────────────────────────┐
│                    FastAPI backend (backend/, :8000)             │
│  scan / preprocess / ingest / dedup / DocScan / settings / tasks │
└──┬──────────────┬──────────────┬──────────────┬─────────────────┘
   │              │              │              │
   ▼              ▼              ▼              ▼
┌─────────┐ ┌──────────┐ ┌─────────────┐ ┌──────────────┐
│  X2MD   │ │  MD2RAG  │ │   ChromaDB  │ │   Ollama     │
│ doc→MD  │ │ chunk→vec│ │  vector_db/  │ │ bge-m3 embed │
│ +chunks │ │  +ingest │ │  per-level   │ │ +opt. rerank │
└────┬────┘ └────┬─────┘ └─────────────┘ └──────────────┘
     │           │
     ▼           ▼
  DOC/<level>/   MD/<level>/
  source docs    .md + .parents/.children.json
```

**Core data flow**:

```
Source docs (DOC/<level>/)
   │  ① Preprocess (X2MD: convert → clean → parent-child chunk)
   ▼
Markdown chunks (MD/<level>/)
   │  ② Ingest (MD2RAG: Ollama bge-m3 embed → ChromaDB)
   ▼
Sensitive-info RAG library (vector_db/, 6 collections: parent+child per level)
   │  ③ Optimization (cross-level dedup, auto-delete high-level duplicates ≥ threshold)
   ▼
   ┌─────────────────────────────────────┐
   │  ④ Test doc → preprocess → embed →   │
   │     hierarchical compare             │ → per-level similarity matches
   └─────────────────────────────────────┘
```

## 🧰 Tech Stack

- **Backend**: Python ≥ 3.9, FastAPI, uvicorn, ChromaDB, sentence-transformers, pdfplumber, python-docx, openpyxl, python-pptx, beautifulsoup4
- **Frontend**: React 18, Ant Design 5, Axios
- **Embedding / LLM**: Ollama (`bge-m3:latest` embedding, optional `qwen3:14b` summaries)
- **Vector DB**: ChromaDB (persisted in `vector_db/`)
- **Optional**: BAAI/bge-reranker-v2-m3 reranker, Tesseract OCR

## 📁 Project Structure

```
ARAG/
├── backend/                # FastAPI backend
│   └── app/
│       ├── main.py         # entry (lifespan inits vector DB)
│       ├── api/endpoints.py# API routes
│       ├── core/config.py  # configuration
│       ├── middleware/auth # API Key auth middleware
│       └── services/       # ingestion / docscan / dedup / preprocess / scanner / reranker ...
├── X2MD/                   # doc → Markdown + parent-child chunking tool
├── MD2RAG/                 # chunk → vector + ingestion tool
├── UI/                     # React frontend
├── DOC/                    # source docs input (by level, runtime data)
│   ├── 0Public/ 1Restricted/ 2Confidential/
├── MD/                     # X2MD output: Markdown + chunk JSON (runtime data)
├── vector_db/              # ChromaDB vector store (per-level, runtime data)
├── DocScan/                # test-doc staging + intermediates (runtime data)
├── dedup_results/          # cross-level dedup results .md / .json (runtime data)
├── bge-m3-local/           # local bge-m3 embedding model (optional)
├── clip-vit-large-patch14-local/  # local CLIP ViT image model (optional)
├── runtime_settings.json   # runtime settings (editable from UI)
├── start.sh / stop.sh      # one-command start / stop scripts (macOS dev)
├── scripts/                # deploy & diagnostic scripts
│   ├── make_deploy_package.sh  # build Linux package (ARAG_V0.2-linux.tar.gz)
│   ├── deploy-dgx.sh · dgx-status.sh  # DGX remote update / status check
│   └── diagnose_ingestion.py etc.  # vector DB / index diagnostics
├── deploy/                 # Linux self-contained package (prod source of truth)
└── DEPENDENCIES.md         # dependency manifest
```

## 🚀 Quick Start

### 1. Requirements

- Python ≥ 3.9
- Node.js ≥ 16
- [Ollama](https://ollama.com) (required for embedding)
- Optional: `antiword` / `libreoffice` (only for `.doc` / `.ppt` conversion)

### 2. Install dependencies

```bash
# Pull embedding model
ollama pull bge-m3:latest

# Optional system deps (macOS)
brew install antiword libreoffice tesseract tesseract-lang

# Backend (X2MD + MD2RAG + backend, editable-installed into backend/.venv)
cd backend && python3 -m venv .venv && source .venv/bin/activate
pip install -e ../X2MD && pip install -e ../MD2RAG && pip install -e .

# Frontend
cd ../UI && npm install
```

### 3. One-command start (dev mode)

```bash
./start.sh            # start backend(:8000) + frontend(:3000)
./start.sh status     # check service status
./start.sh stop        # stop all services
# Or start individually:
./start.sh back       # backend only
./start.sh front      # frontend only
```

Then visit:
- Frontend UI: http://localhost:3000
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## 📖 Usage Workflow

1. **Place source docs**: put sensitive documents into `DOC/<level>/` (`0Public` / `1Restricted` / `2Confidential`).
2. **Preprocess**: trigger preprocessing from the UI (X2MD conversion + parent-child chunking) → `MD/<level>/`.
3. **Ingest**: vectorize via Ollama bge-m3 into ChromaDB to build the sensitive-info RAG library.
4. **Optimize (optional)**: cross-level dedup, deleting child chunks duplicated from lower levels in higher-level libraries.
5. **Document scan**: in "File Scan", upload a test doc → preprocess → embed → compare to view per-level similarity matches; or paste text in "Text Scan".

## 🔌 API Overview

All endpoints are prefixed with `/api/v1`; interactive docs at `/docs`.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/scan/file` | Upload a test file to DocScan |
| POST | `/scan/text` | Text sensitive-info scan |
| POST | `/docscan/preprocess` · `/embed` · `/compare` | DocScan preprocess / embed / hierarchical compare |
| GET | `/docscan/files` · `/status` · `/stats` | DocScan file list / status / stats |
| POST | `/preprocess` · `/preprocess/retry` | Preprocess / retry failures |
| POST | `/tasks/preprocess` · `/tasks/ingest` · `/tasks/dedup` | Async background tasks (progress / cancellable) |
| POST | `/ingest` · DELETE `/reset` | Ingest / reset vector DB |
| POST | `/dedup` · `/dedup/apply` · GET `/dedup/results` | Dedup compare / apply / view results |
| GET · DELETE | `/dedup/results/file` | View / delete a single dedup result |
| GET | `/scan-source-dir` | Scan source dir (DOC/level) file stats |
| GET | `/stats` · `/status` | Vector DB stats / ingestion status |
| GET · PUT · POST | `/settings` · `/settings/{category}/reset` | Read / update / reset runtime config |
| GET | `/tasks/active` · `/tasks/latest/{type}` · `/tasks/{id}` · POST `/tasks/{id}/cancel` | Task query & cancellation |

## ⚙️ Configuration

- **`backend/.env`**: `API_KEY` (auth), `AUTH_DISABLED` (dev no-auth).
- **`runtime_settings.json`**: similarity thresholds (currently confidential 0.85 / restricted 0.7), chunk params, upload size, embedding model — editable from the UI Settings page.
- **`X2MD/x2md.conf`**: conversion / chunking / LLM / ViT config.
- **`MD2RAG/md2rag.conf`**: embedding model, vector DB path, device (`[device] type=cpu` forces Ollama).

**Similarity tiers** (dedup/compare): `≥0.8` high / `0.65–0.8` medium / `0.5–0.65` weak.

## 📦 Deployment (Linux / DGX Spark)

A self-contained deploy package is provided (`ARAG_V0.2-linux.tar.gz` + `deploy/`):

```bash
tar xzf ARAG_V0.2-linux.tar.gz
cd ARAG_V0.2
./setup.sh            # auto-install system deps / venv / Python deps / ollama pull bge-m3
./start.sh            # single uvicorn serves UI(/) + API(/api/v1), default 0.0.0.0:8000
ARAG_HOST=0.0.0.0 ARAG_PORT=9000 ./start.sh   # custom port
```

**Remote update of an existing install (DGX Spark etc.)**:

```bash
# Build on your machine
npm --prefix UI run build                          # build frontend (relative /api/v1, same-origin)
./scripts/make_deploy_package.sh                   # produce ARAG_V0.2-linux.tar.gz
scp ARAG_V0.2-linux.tar.gz scripts/deploy-dgx.sh scripts/dgx-status.sh alpha@<DGX-IP>:~/
# On the DGX (password prompt)
bash deploy-dgx.sh    # atomic switch: keeps data/settings/venv, auto-rollback on failure
bash dgx-status.sh    # check stage/bak/venv/build/port/process/log
```

See [`deploy/README.md`](deploy/README.md) for details.

## 🔒 Security Notes

- Dev / deploy packages default to no auth (`deploy/setup.sh` writes `AUTH_DISABLED=True`; `app/core/config.py` defaults to `False` and must be explicitly enabled) — for convenient intranet deployment only — **in production always set `API_KEY` and enable authentication**.
- The system may handle classified documents; `DOC/`, `MD/`, `vector_db/`, `DocScan/`, `dedup_results/` contain plaintext — enforce disk access control.
- All file-handling endpoints include path-traversal defenses (basename validation + post-`resolve` containment check).
