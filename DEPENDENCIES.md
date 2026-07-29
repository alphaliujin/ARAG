# MD2RAG 项目依赖清单

## 系统依赖

### macOS
```bash
brew install antiword libreoffice
```

### Ubuntu/Debian
```bash
apt-get install antiword libreoffice
```

## Python 依赖

### 核心依赖（后端）
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pdfplumber>=0.10.0
python-docx>=1.0.0
openpyxl>=3.1.0
python-pptx>=0.6.0
beautifulsoup4>=4.12.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
numpy>=1.24.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0
aiofiles>=23.0.0
```

### X2MD 额外依赖
```
# 老版本 Office 文件支持
xlrd>=2.0.0          # .xls 文件支持

# 可选：OCR 支持
pytesseract>=0.3.10  # 图片 OCR
pillow>=10.0.0       # 图像处理

# 可选：ViT 图像特征
transformers>=4.30.0
torch>=2.0.0

# 可选：LLM 增强
# Ollama 本地运行，无需额外包
```

### 前端依赖
```
react>=18.2.0
react-dom>=18.2.0
react-scripts@5.0.1
antd>=5.12.0
@ant-design/icons>=5.2.6
axios>=1.6.2
```

## 开发依赖
```
pytest>=7.0.0
ruff>=0.4.0
```

---

## 安装命令

### 一次性安装所有依赖
```bash
# macOS
brew install antiword libreoffice

# 项目 Python 依赖
cd X2MD && pip install -e .
cd ../backend && pip install -e .

# 前端依赖
cd ../UI && npm install
```

---

## 版本信息
- Python: >=3.9
- Node.js: >=16

## 备注
- xlrd 2.0+ 仅支持 .xls，不支持 .xlsx（使用 openpyxl 处理 .xlsx）
- antiword 用于快速转换 .doc，LibreOffice 作为备用
- LibreOffice 也用于转换 .ppt 到 .pptx
