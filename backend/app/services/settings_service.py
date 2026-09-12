"""设置管理服务 - 持久化运行时设置到 JSON 文件.

支持的设置类别:
- general: 通用参数（相似度阈值、上传大小、日志等）
- model: 模型配置（嵌入模型、Ollama 配置等）
- preprocess: 预处理配置（切片大小、OCR、LLM等）
- database: 数据库配置（向量目录、集合前缀等）
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings


# 设置文件路径
SETTINGS_FILE = Path(settings.VECTOR_DB_DIR).parent / "runtime_settings.json"

# 默认设置（与前端 initialValues 对齐）
DEFAULT_SETTINGS = {
    "general": {
        "similarityThresholdConfidential": settings.SIMILARITY_THRESHOLD_CONFIDENTIAL,
        "similarityThresholdRestricted": settings.SIMILARITY_THRESHOLD_RESTRICTED,
        "maxUploadSize": settings.MAX_UPLOAD_SIZE // (1024 * 1024),  # MB
        "enableLogging": True,
        "logLevel": "info",
    },
    "model": {
        "embeddingModel": settings.EMBEDDING_MODEL,
        "ollamaUrl": settings.OLLAMA_BASE_URL,
        "ollamaModel": "bge-m3:latest",
        "stModel": "all-MiniLM-L6-v2",
        "batchSize": settings.EMBEDDING_BATCH_SIZE,
    },
    "preprocess": {
        "chunkSize": 500,
        # 与 backend config.CHUNK_OVERLAP_DEFAULT (=100, A4 修复) 及 X2MD x2md.conf
        # (chunk_overlap=100) 对齐。原值 50 会在启动时经 _apply_runtime_settings
        # 覆盖 config 的 100, 静默回退 A4 修复, 导致句子边界处敏感信息硬切。
        "chunkOverlap": 100,
        "enableOCR": True,
        "enableLLM": False,
        "ocrLang": "chi_sim+eng",
        "extractTables": True,
    },
    "database": {
        "vectorDbDir": settings.VECTOR_DB_DIR,
        "collectionPrefix": "md2rag",
        "anonymizedTelemetry": False,
    },
}


def _validate_ollama_url(url: Any) -> str:
    """校验 ollamaUrl, 防止 SSRF.

    嵌入/LLM 调用时文档原文会 POST 到该 URL, 必须限制为 http/https 且拒绝云元数据
    等危险端点。允许 loopback / RFC1918 内网 (Ollama 常部署于本地或局域网)。

    除字面主机名黑名单外, 再把 host 经 getaddrinfo 解析成实际 IP 逐条校验,
    拦截编码形式的绕过 (IPv6-mapped / 十六进制 / 十进制整数 / 短别名):
    这些表示经系统解析后归一到同一 IP, 只要任一解析结果落在 link-local
    (169.254.0.0/16、fe80::/10, 云元数据端点所在段) 即拒绝。解析失败
    (如仅 /etc/hosts 或 mDNS 可解析的局域网别名) 不误拒 — 请求侧同样不可达。
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    raw = str(url)
    try:
        parsed = urlparse(raw)
    except Exception as e:
        raise ValueError(f"ollamaUrl is not a valid URL: {raw!r} ({e})")
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"ollamaUrl scheme must be http or https, got: {scheme!r} (url={raw!r})"
        )
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError(f"ollamaUrl must have a host: {raw!r}")
    # 域名层面的已知元数据端点兜底 (即使本机 DNS 解析不到也被拦截)。
    if host in ("169.254.169.254", "metadata.google.internal", "metadata"):
        raise ValueError(
            f"ollamaUrl points to a link-local/metadata endpoint, blocked for SSRF protection: {raw!r}"
        )
    # 解析 host 为实际 IP 再校验: 拦截编码形式 (IPv6-mapped/十六进制/十进制整数)。
    # DNS rebinding 无法在校验时刻完全消除, 但可显著收窄攻击面。
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return raw  # 不可解析 (局域网别名等): 不误拒, 请求侧同样会失败
    for info in infos:
        ip_str = info[4][0].split("%")[0]  # 去掉 IPv6 scope id (fe80::1%eth0)
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        # IPv4-mapped IPv6 (::ffff:a.b.c.d) 归一回 IPv4 再判
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if ip.is_link_local:
            raise ValueError(
                f"ollamaUrl resolves to link-local address {ip}, blocked for SSRF protection: {raw!r}"
            )
    return raw


class SettingsService:
    """设置管理服务 - 线程安全."""

    def __init__(self):
        # RLock：update_category() 持锁后会再调用 _apply_runtime_settings()，
        # 后者也需要进入临界区。非重入锁会在嵌套 with 时永久阻塞，
        # 触发 PUT /settings/database 等接口挂死（P0）。
        self._lock = threading.RLock()
        self._settings: Optional[Dict[str, Any]] = None

    def _load_from_file(self) -> Dict[str, Any]:
        """从文件加载设置，文件不存在时返回默认值.

        损坏的 JSON 会被重命名为 .broken.<ts> 备份,避免用户的合法编辑
        因下一次 save 被静默覆盖,同时在控制台留下 ERROR 级提示。
        """
        if not SETTINGS_FILE.exists():
            return self._deep_copy(DEFAULT_SETTINGS)

        try:
            with SETTINGS_FILE.open("r", encoding="utf-8") as f:
                loaded = json.load(f)

            # 合并默认值，确保新增的字段也有默认值
            merged = self._deep_copy(DEFAULT_SETTINGS)
            for category, values in loaded.items():
                if category in merged and isinstance(values, dict):
                    merged[category].update(values)
                else:
                    merged[category] = values
            return merged
        except json.JSONDecodeError as e:
            # 损坏的 JSON: 移到 .broken.<ts> 备份,防止后续 save 覆盖用户原始内容
            import time as _time
            backup = SETTINGS_FILE.with_suffix(f".json.broken.{int(_time.time())}")
            try:
                SETTINGS_FILE.rename(backup)
                print(f"[SETTINGS] ERROR malformed JSON in {SETTINGS_FILE}: {e}; backed up to {backup.name}, using defaults")
            except OSError as rename_err:
                print(f"[SETTINGS] ERROR malformed JSON in {SETTINGS_FILE}: {e} (rename to backup failed: {rename_err}); using defaults")
            return self._deep_copy(DEFAULT_SETTINGS)
        except OSError as e:
            print(f"[SETTINGS] Error reading settings file: {e}, using defaults")
            return self._deep_copy(DEFAULT_SETTINGS)

    def _save_to_file(self, data: Dict[str, Any]) -> None:
        """保存设置到文件."""
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SETTINGS_FILE.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _deep_copy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """深拷贝."""
        return copy.deepcopy(data)

    def get_all(self) -> Dict[str, Any]:
        """获取所有设置."""
        with self._lock:
            if self._settings is None:
                self._settings = self._load_from_file()
            return self._deep_copy(self._settings)

    def get_category(self, category: str) -> Dict[str, Any]:
        """获取指定类别的设置."""
        all_settings = self.get_all()
        return all_settings.get(category, {})

    # 设置校验规则: key → (min, max, 类型)
    _VALIDATION_RULES = {
        "similarityThresholdConfidential": (0.0, 1.0, float),
        "similarityThresholdRestricted": (0.0, 1.0, float),
        "maxUploadSize": (1, 500, int),
        "batchSize": (1, 128, int),
        "chunkSize": (100, 5000, int),
        "chunkOverlap": (0, 500, int),
    }

    # 嵌入模型允许值列表 (不在 _VALIDATION_RULES 中, 因为不适用 min/max/type 模式)
    _ALLOWED_EMBEDDING_MODELS = ("ollama-bge-m3", "mps-bge-m3")

    # OCR 语言允许值 (与前端 SettingsPage 下拉一致)
    _ALLOWED_OCR_LANGS = ("chi_sim+eng", "chi_sim", "eng")

    def update_category(self, category: str, values: Dict[str, Any]) -> Dict[str, Any]:
        """更新指定类别的设置，返回更新后的该类别设置."""
        if category not in DEFAULT_SETTINGS:
            raise ValueError(f"Unknown settings category: {category}")

        # 校验值范围和类型
        for key in list(values.keys()):
            val = values[key]
            if key in self._VALIDATION_RULES:
                min_val, max_val, expected_type = self._VALIDATION_RULES[key]
                try:
                    converted = expected_type(val)
                except (ValueError, TypeError):
                    raise ValueError(f"Setting '{key}' must be {expected_type.__name__}, got: {val}")
                if not (min_val <= converted <= max_val):
                    raise ValueError(f"Setting '{key}' must be between {min_val} and {max_val}, got: {converted}")
                # 存转换后的值, 避免 "10"/1.9 经校验通过却以原始类型持久化/返回
                # (旧实现存原始 val, get_all 返回 "10" 字符串或 1.9 浮点, 与校验类型不符)
                values[key] = converted

        # 校验 embeddingModel 特殊规则
        if "embeddingModel" in values and values["embeddingModel"] not in self._ALLOWED_EMBEDDING_MODELS:
            raise ValueError(
                f"Setting 'embeddingModel' must be one of {self._ALLOWED_EMBEDDING_MODELS}, got: {values['embeddingModel']}"
            )

        # 校验 ocrLang 取值
        if "ocrLang" in values and values["ocrLang"] not in self._ALLOWED_OCR_LANGS:
            raise ValueError(
                f"Setting 'ocrLang' must be one of {self._ALLOWED_OCR_LANGS}, got: {values['ocrLang']}"
            )

        # 校验 ollamaUrl (SSRF 防护: 限制 scheme + 拒绝云元数据端点)
        if "ollamaUrl" in values:
            values["ollamaUrl"] = _validate_ollama_url(values["ollamaUrl"])

        # 拒绝不在 _VALIDATION_RULES 或 DEFAULT_SETTINGS[category] 中的未知 key
        if category in DEFAULT_SETTINGS:
            allowed_keys = set(DEFAULT_SETTINGS[category].keys())
            unknown_keys = set(values.keys()) - allowed_keys
            if unknown_keys:
                raise ValueError(f"Unknown settings keys for '{category}': {unknown_keys}. Allowed: {allowed_keys}")

        with self._lock:
            if self._settings is None:
                self._settings = self._load_from_file()

            # 合并到现有类别
            if category not in self._settings:
                self._settings[category] = {}
            self._settings[category].update(values)

            # 持久化
            self._save_to_file(self._settings)

            # 应用到运行时（部分关键设置实时生效）
            self._apply_runtime_settings(category, values)

            return self._deep_copy(self._settings[category])

    def reset_category(self, category: str) -> Dict[str, Any]:
        """重置指定类别为默认值."""
        if category not in DEFAULT_SETTINGS:
            raise ValueError(f"Unknown settings category: {category}")

        with self._lock:
            if self._settings is None:
                self._settings = self._load_from_file()

            self._settings[category] = self._deep_copy(DEFAULT_SETTINGS[category])
            self._save_to_file(self._settings)
            self._apply_runtime_settings(category, self._settings[category])
            return self._deep_copy(self._settings[category])

    def _apply_runtime_settings(self, category: str, values: Dict[str, Any]) -> None:
        """将部分关键设置应用到 settings 对象（运行时生效）.

        ★ database 类的 vectorDbDir 和 collectionPrefix 需要重启后生效,
          因为 ChromaDB PersistentClient 在初始化时就锁定了路径和集合名,
          运行时修改路径会导致客户端指向不一致的位置。
          此处仅做防御性校验: 确保路径为绝对路径, 防止相对路径溜进来。
        """
        if category == "general":
            if "similarityThresholdConfidential" in values:
                settings.SIMILARITY_THRESHOLD_CONFIDENTIAL = float(values["similarityThresholdConfidential"])
            if "similarityThresholdRestricted" in values:
                settings.SIMILARITY_THRESHOLD_RESTRICTED = float(values["similarityThresholdRestricted"])
            if "maxUploadSize" in values:
                settings.MAX_UPLOAD_SIZE = int(values["maxUploadSize"]) * 1024 * 1024
        elif category == "model":
            if "embeddingModel" in values:
                settings.EMBEDDING_MODEL = str(values["embeddingModel"])
            if "ollamaUrl" in values:
                settings.OLLAMA_BASE_URL = str(values["ollamaUrl"])
            if "batchSize" in values:
                settings.EMBEDDING_BATCH_SIZE = int(values["batchSize"])
        elif category == "preprocess":
            # 预处理默认值实时生效: 服务层在请求体字段为 None 时回退到这里
            # (UI/PreprocessPage.js 也会在挂载时 GET /settings/preprocess 同步表单初值)
            if "chunkSize" in values:
                settings.CHUNK_SIZE_DEFAULT = int(values["chunkSize"])
            if "chunkOverlap" in values:
                settings.CHUNK_OVERLAP_DEFAULT = int(values["chunkOverlap"])
            if "extractTables" in values:
                settings.EXTRACT_TABLES_DEFAULT = bool(values["extractTables"])
            if "enableLLM" in values:
                settings.ENABLE_LLM_DEFAULT = bool(values["enableLLM"])
            if "enableOCR" in values:
                # 注: 当前 X2MD CLI 无 --no-ocr 开关, 仅持久化, 不影响实际处理
                settings.ENABLE_OCR_DEFAULT = bool(values["enableOCR"])
            if "ocrLang" in values:
                settings.OCR_LANG_DEFAULT = str(values["ocrLang"])
        elif category == "database":
            # ★ 防御性校验: vectorDbDir 必须是绝对路径
            # 相对路径会随 os.getcwd() 变化, 导致 ChromaDB 数据碎片化或丢失
            if "vectorDbDir" in values:
                path_val = str(values["vectorDbDir"])
                if not os.path.isabs(path_val):
                    # 自动转换为绝对路径 (相对于项目根目录)
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                    path_val = os.path.normpath(os.path.join(project_root, path_val))
                    # 同步修正 _settings 中的值, 避下次加载时仍是相对路径
                    with self._lock:
                        if self._settings and "database" in self._settings:
                            self._settings["database"]["vectorDbDir"] = path_val
                            self._save_to_file(self._settings)
                    print(f"[SETTINGS] vectorDbDir: relative path '{values['vectorDbDir']}' → absolute '{path_val}'")


settings_service = SettingsService()

# 启动时加载并应用持久化设置
try:
    _loaded = settings_service.get_all()
    for _cat, _vals in _loaded.items():
        settings_service._apply_runtime_settings(_cat, _vals)
except Exception as _e:
    print(f"[SETTINGS] Failed to apply persisted settings: {_e}")
