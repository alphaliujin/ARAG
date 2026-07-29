# ARAG_V0.2 部署说明 (Linux)

本包为**自包含部署包**: 单进程 uvicorn 同时服务前端(`/`)和 API(`/api/v1`),
无需 nginx / Node。嵌入走 Ollama(bge-m3),不依赖本地 PyTorch 模型。

## 一、目标机器要求

| 项 | 要求 |
|---|---|
| OS | Linux (x86_64 / arm64),内核较新即可 |
| Python | >= 3.9 (缺失时 setup.sh 自动安装) |
| Ollama | 可选 (缺失时 setup.sh 自动安装并启动服务) |
| 可选 | `antiword` / `libreoffice` (仅 .doc/.ppt 转换需要; docx/pdf/xlsx/pptx/html/md 不需要) |

内存建议 >= 8GB(嵌入时 ChromaDB + Ollama 同时占用)。磁盘: 依赖安装约 3-4GB(含 torch/chromadb),
Ollama bge-m3 模型约 1.2GB。

## 二、安装(解压后执行一次)

```bash
tar xzf ARAG_V0.2-linux.tar.gz
cd ARAG_V0.2
./setup.sh
```

`setup.sh` 会: 检测并自动安装系统依赖(python3/pip3/python3-venv/curl/Ollama, 缺失才装) → 建 venv → 装 Python 依赖(首次较慢) → `ollama pull bge-m3:latest` →
把配置里的 `__INSTALL_DIR__` 替换为实际路径 → 建数据目录骨架。

## 三、启动 / 停止

```bash
./start.sh          # 启动 (默认 0.0.0.0:8000, 单进程服务 UI + API)
./start.sh status    # 查看状态
./start.sh stop      # 停止
# 自定义端口/地址:
ARAG_HOST=0.0.0.0 ARAG_PORT=9000 ./start.sh
```

访问:
- 前端界面: http://<机器IP>:8000
- API 文档: http://<机器IP>:8000/docs
- 健康检查: http://<机器IP>:8000/health

日志: `tail -f server.log`

## 四、使用流程

1. 把待处理文档放入 `DOC/<密级>/`(`0Public` / `1Restricted` / `2Confidential`)。
2. 浏览器打开前端 → 「文件扫描」上传比对文件 → 预处理 → 生成向量 → 比对;
   或「文本扫描」直接粘贴文本检测敏感信息。
3. 数据入库后可「数据优化」(跨密级去重)。

## 五、配置说明

- `backend/.env`: 默认 `AUTH_DISABLED=True`(免认证, 仅限内网/开发)。
  生产部署请设 `API_KEY=xxx` 并删除 `AUTH_DISABLED`(或设 False),前端构建时注入 `REACT_APP_API_KEY`。
- `X2MD/x2md.conf`、`MD2RAG/md2rag.conf`、`runtime_settings.json`: 已由 `setup.sh` 写入实际路径。
  运行时后端会用 `app/core/config.py`(基于安装位置派生)覆盖关键路径,故移动安装目录后重启即可。
- `runtime_settings.json` 可在前端「设置」页调整(相似度阈值、切片参数、上传大小等)。

## 六、架构

```
浏览器 ──> :8000 uvicorn (单进程)
              ├── /api/v1/*  → FastAPI 路由 (扫描/预处理/入库/去重/DocScan)
              ├── /health, /docs
              └── /          → UI/build (前端静态, 同源, 走 /api/v1 调后端)
后端依赖:
  ├── ChromaDB  (vector_db/, 持久化向量)
  ├── Ollama    (localhost:11434, bge-m3 嵌入)
  └── X2MD / MD2RAG (经 _invoke.py + sys.path 导入源码)
```

## 七、常见问题

- **`ollama pull` 失败**: 先 `ollama serve` 启动服务,再 `ollama pull bge-m3:latest`。
- **端口被占**: `ARAG_PORT=其它端口 ./start.sh`。
- **.doc 转换失败**: 装 `antiword` 或 `libreoffice`。
- **改了 Python 代码**: `./start.sh stop && ./start.sh` 重启即生效(uvicicorn 非 reload 模式)。
- **迁移安装目录**: 直接移动整个目录后重启即可(路径由安装位置派生)。

## 八、安全提醒

- 默认 `AUTH_DISABLED=True` 仅为方便内网部署, **生产环境务必设 `API_KEY` 并启用认证**。
- 系统处理的可能是机密文档, 部署机器的 `DOC/`、`MD/`、`vector_db/`、`DocScan/`、`dedup_results/` 含明文内容, 注意磁盘访问控制。


## 九、DGX Spark 部署说明

DGX Spark (NVIDIA GH200 Grace Hopper, ARM64/aarch64, Ubuntu + NVIDIA GPU) 是本系统的典型部署目标。
源码部署包架构无关, 在 DGX Spark 上原生可用, 注意以下几点:

- **架构 aarch64**: `setup.sh` 会装 aarch64 wheel (fastapi/chromadb/numpy/pdfplumber 等均有 aarch64 wheel);
  `sentence-transformers` 会拉 torch (aarch64 wheel, 体积较大, 首次安装慢)。本系统**实际嵌入走 Ollama**,
  torch 仅因 sentence-transformers 被装上, 不影响运行。
- **GPU 加速**: 确保 NVIDIA 驱动 + CUDA 已装 (DGX Spark 出厂自带)。`setup.sh` 会检测 `nvidia-smi`;
  Ollama 自动用 GPU 跑 bge-m3, 嵌入速度远快于 CPU。
- **Ollama**: setup.sh 检测到缺失会自动用官方脚本安装 (支持 aarch64) 并启动服务, 无需手动。
- **内存**: DGX Spark 统一内存 128GB/288GB, 远超本系统需要 (ChromaDB + Ollama 同占通常 < 8GB)。
- **端口检测**: `start.sh` 的 stop/status 已兼容无 `lsof` 的 Ubuntu (回退 `ss`/`fuser`)。
- **绑定地址**: 默认 `0.0.0.0:8000`, 局域网内可直接访问前端。如需仅本机访问: `ARAG_HOST=127.0.0.1 ./start.sh`。
- **认证**: 默认 `AUTH_DISABLED=True` 仅限内网。DGX Spark 上跑机密文档时, 务必在 `backend/.env` 设
  `API_KEY=xxx` 并删除/置 False `AUTH_DISABLED`, 前端构建时注入 `REACT_APP_API_KEY`。
