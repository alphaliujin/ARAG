"""API Key 认证中间件.

用法: 在 Settings 中设置 API_KEY (通过 .env 或环境变量).
- API_KEY 非空: 所有请求需携带 X-API-Key header
- API_KEY 为空 + AUTH_DISABLED=True: 无认证 (仅限开发/内网环境)
- API_KEY 为空 + AUTH_DISABLED=False: 拒绝所有请求（防止生产环境无认证）

免认证路由:
- GET /          (根信息)
- GET /health    (健康检查)

⚠️ 当 AUTH_DISABLED=True 时，启动时会打印警告日志，提醒操作者认证已禁用。
"""

import hmac
import logging

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.config import settings

logger = logging.getLogger("arag.auth")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """检查请求是否携带有效 API Key."""

    # 不需要认证的路径前缀。
    # 注: /docs /openapi.json /redoc 不再免认证 - 暴露完整 OpenAPI schema 会泄露
    # /reset /dedup/apply 等敏感路由, 利于侦察。开发期 AUTH_DISABLED=True 时仍可访问。
    PUBLIC_PATHS = ("/", "/health")

    _startup_warned = False

    async def dispatch(self, request: Request, call_next):
        # 未配置 API_KEY → 检查 AUTH_DISABLED 标志
        if not settings.API_KEY:
            if not settings.AUTH_DISABLED:
                # API_KEY 未配置且未显式禁用认证 → 拒绝所有请求
                raise HTTPException(
                    status_code=500,
                    detail="API_KEY not configured. Set API_KEY in environment/.env, or set AUTH_DISABLED=True for development.",
                )
            if not self._startup_warned:
                logger.warning(
                    "⚠️  AUTH_DISABLED=True — authentication is DISABLED. "
                    "All requests will be accepted without an API key. "
                    "Set API_KEY in your environment or .env file before deploying to production."
                )
                self._startup_warned = True
            return await call_next(request)

        # 公开路由免认证
        path = request.url.path
        if path in self.PUBLIC_PATHS:
            return await call_next(request)

        # CORS 预检 (OPTIONS) 不携带自定义请求头 (含 X-API-Key),
        # 必须放行交由内层 CORSMiddleware 处理, 否则浏览器跨域 POST 的预检
        # 会被这里 401 拦截, 导致整个前端在启用认证后跨域请求全部失败。
        if request.method == "OPTIONS":
            return await call_next(request)

        # 仅从 header 取 key (不再接受 query param，避免泄露到 URL/日志)
        api_key = request.headers.get("X-API-Key", "")

        if not hmac.compare_digest(api_key, settings.API_KEY):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API Key. Pass via X-API-Key header.",
            )

        return await call_next(request)
