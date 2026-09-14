import sys
from pathlib import Path

# 添加 MD2RAG 到 Python 路径
# main.py 位于 backend/app/main.py
# MD2RAG 位于项目根目录下，与 backend 同级
# 从 main.py 出发: backend(1) -> ARAG_V0.2(2) -> MD2RAG
MD2RAG_PATH = Path(__file__).resolve().parent.parent.parent / "MD2RAG"
if str(MD2RAG_PATH) not in sys.path:
    sys.path.insert(0, str(MD2RAG_PATH))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 关键: settings_service 必须在 vector_db_service 之前导入。
# settings_service 在 module-load 时会读 runtime_settings.json 并把 EMBEDDING_MODEL 等
# 写回 app.core.config.settings 单例;而 vector_db_service.initialize() 在 lifespan
# 中根据 settings.EMBEDDING_MODEL 选择 embedder。若顺序颠倒,启动时永远拿到 .env 默认值。
from app.services import settings_service as _settings_service_init  # noqa: F401

from app.api.endpoints import router
from app.core.config import settings
from app.middleware.auth import APIKeyMiddleware
from app.services.vector_db import vector_db_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 初始化向量数据库
    vector_db_service.initialize()
    yield
    # Shutdown: 清理资源 (如有需要可在此扩展)


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="定密审核系统 - 基于向量数据库的文档涉密/机密信息检测",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # 显式列 origin (见 config.CORS_ORIGINS); 不再用 "*" + credentials=True 的非法组合
    # 那种组合下浏览器会拒绝响应, Starlette 也会静默把 credentials 关掉, 不如显式
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*", "X-API-Key"],
)

# API Key 认证中间件 (API_KEY 为空时自动跳过, 不影响开发调试)
app.add_middleware(APIKeyMiddleware)

app.include_router(router, prefix=settings.API_PREFIX)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# 部署模式: 若前端构建产物(UI/build)存在, 后端统一服务 UI(/) + API(/api/v1),
# 单进程同源, 无需 nginx/node。/api/v1/* 与 /health 优先于该 mount(已先注册)。
# 开发模式(UI/build 不存在)退化为原 JSON 根信息。
_build_dir = Path(__file__).resolve().parent.parent.parent / "UI" / "build"
if _build_dir.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_build_dir), html=True), name="ui")
else:
    @app.get("/")
    async def root():
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "running"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=True
    )