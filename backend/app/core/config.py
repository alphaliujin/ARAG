from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "敏感信息识别系统"
    VERSION: str = "0.2.0"
    API_PREFIX: str = "/api/v1"

    BACKEND_HOST: str = "127.0.0.1"  # 默认仅本机访问; 需对外暴露时可改为 0.0.0.0
    BACKEND_PORT: int = 8000

    # CORS 允许的前端 origin (CRA 默认 3000, 127.0.0.1 是 Safari 必备)
    # 注意: 不能同时用 ["*"] + allow_credentials=True (浏览器拒绝); main.py 也据此关掉了 credentials
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    DATA_DIR: str = os.path.join(_BASE_DIR, "MD")
    # 密级目录名常量统一来自 md2rag.loader.CLASSIFICATION_DIR_MAP,
    # 这里不再保留 PUBLIC_DIR / CONFIDENTIAL_DIR / RESTRICTED_DIR / SECRET_DIR 副本。

    VECTOR_DB_DIR: str = os.path.join(_BASE_DIR, "vector_db")

    # 嵌入模型配置: 支持 ollama-bge-m3 (默认) 和 mps-bge-m3, 二者均 1024维.
    # 同一向量库内必须前后使用同一 embedder, 切换模型需要清空向量库重建,
    # 否则 sentence-transformers (MPS) 与 Ollama 服务端的数值实现差异会让
    # 检索 ranking 出现偏差 (维度相同, 数值不完全一致).
    EMBEDDING_MODEL: str = "ollama-bge-m3"
    EMBEDDING_BATCH_SIZE: int = 32
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # 计算设备配置: auto/mps/cuda/cpu (仅在选择 mps-bge-m3 或本地模型时使用)
    DEVICE: str = "auto"

    # Cross-encoder reranker (B2): 可选, 留空则 DocScan 走传统 combined_similarity
    # 期望模型 BAAI/bge-reranker-v2-m3, 本地目录路径 (空 = 用项目默认 bge-reranker-v2-m3-local/)
    # 模型不存在时优雅降级, 不阻塞主流程
    RERANKER_MODEL_PATH: str = ""

    SIMILARITY_THRESHOLD_CONFIDENTIAL: float = 0.6
    SIMILARITY_THRESHOLD_RESTRICTED: float = 0.7

    # 预处理默认参数 (X2MD 切片/OCR/LLM 默认值, 由 SettingsService 在运行时覆盖).
    # 调用方在请求体里传 None 表示"用默认", 服务层据此回退到这里, 实现 Settings 联动。
    CHUNK_SIZE_DEFAULT: int = 500
    # A4: 0 → 100, 让 child_overlap (cli.py 派生 max(eff//2, 60)) 不再太小,
    # 减少"句子被硬切到两个 chunk"导致两侧向量都不全的漏检
    CHUNK_OVERLAP_DEFAULT: int = 100
    EXTRACT_TABLES_DEFAULT: bool = True
    ENABLE_LLM_DEFAULT: bool = False
    ENABLE_OCR_DEFAULT: bool = True            # 当前 X2MD CLI 无 --no-ocr 开关, 仅持久化保留
    OCR_LANG_DEFAULT: str = "chi_sim+eng"      # 透传到 X2MD --ocr-lang

    DOCSCAN_DIR: str = os.path.join(_BASE_DIR, "DocScan")

    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024

    # API 认证: 设置 API_KEY 后启用认证中间件; 空字符串 = 需 AUTH_DISABLED=True 才能跳过
    API_KEY: str = ""
    AUTH_DISABLED: bool = False  # 必须显式 True 才能在 API_KEY 为空时绕过认证(仅限开发)

    model_config = {"env_file": ".env", "case_sensitive": True}

settings = Settings()