"""日志配置模块 - 用于记录 MD2RAG 详细运行日志."""

import logging
import logging.handlers
import sys
import os
import time
import traceback
from pathlib import Path
from datetime import datetime


class _MillisecondFormatter(logging.Formatter):
    """Formatter that appends milliseconds to the timestamp.

    Standard `logging.Formatter.formatTime` uses `time.strftime`, which does NOT
    support `%f` (microseconds). Naively putting `%f` in `datefmt` results in
    a literal `f` on macOS (and `%f` on some Linux variants). We instead format
    seconds with strftime and append `.NNN` (milliseconds) manually.
    """
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        if datefmt:
            # Strip any trailing .%f the caller may have left in (legacy)
            clean_fmt = datefmt.replace('.%f', '').replace('%f', '')
            s = time.strftime(clean_fmt, ct)
        else:
            s = time.strftime("%Y-%m-%d %H:%M:%S", ct)
        return f"{s}.{int(record.msecs):03d}"

# 日志文件路径 - 写入到 MD2RAG 目录根目录
LOG_FILE = Path(__file__).parent.parent / "MD2RAG.log"

# 创建格式化器 — 用自定义 Formatter 正确展开毫秒(标准 datefmt 不支持 %f)
formatter = _MillisecondFormatter(
    '[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S.%f'
)

# 创建文件处理器 — 用 RotatingFileHandler 限制单个日志文件 ≤20MB,保留 5 个备份
# 旧实现 FileHandler 无轮转,日志会无限增长直到填满磁盘。
try:
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        mode='a',
        maxBytes=20 * 1024 * 1024,  # 20 MB
        backupCount=5,
        encoding='utf-8',
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
except (OSError, IOError) as e:
    # 如果文件无法写入（权限/磁盘空间），退回到仅控制台日志
    print(f"[WARNING] 无法创建日志文件 {LOG_FILE}: {e}", file=sys.stderr)
    file_handler = None

# 创建控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)


# 已为某 logger 注册过 handler 的标记。用 name 而非 id():
#   - id() 在 logger 被 GC 后会被复用,可能误判已注册;
#   - logger.getLogger(name) 已对相同 name 缓存同一实例,name 集合更稳定。
_registered: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """获取配置好的日志记录器."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加处理器 — 用 name 作 key,无 GC 误用问题
    if name not in _registered:
        if file_handler:
            logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # 关闭向上传播: 本模块给每个 md2rag.* logger 都挂了 handler,
        # 若 propagate=True, 子 logger(如 md2rag.loader)的记录会再传给
        # 同样挂了 handler 的根 md2rag logger, 导致每条日志输出两次
        # (stdout 与 MD2RAG.log 各翻倍)。挂 handler 的 logger 不再传播即可。
        logger.propagate = False
        _registered.add(name)

    return logger


def setup_exception_logging():
    """设置全局未捕获异常处理器，确保崩溃时留下日志."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        # 获取根日志器
        logger = get_logger("md2rag.uncaught")
        # 格式化异常信息
        exc_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        exc_text = "".join(exc_lines)
        logger.critical(f"未捕获的异常 ({exc_type.__name__}): {exc_value}\n{exc_text}")
        # 同时输出到 stderr
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    sys.excepthook = handle_exception


def log_step(logger: logging.Logger, step_name: str, details: str = ""):
    """记录步骤日志.

    当 logger 级别 < INFO 时快速返回（避免 f-string 与调用开销）。
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    msg = f"[STEP] {step_name}"
    if details:
        msg += f" | {details}"
    logger.info(msg)


def log_timing(logger: logging.Logger, operation: str, elapsed_ms: float):
    """记录耗时日志.

    当 logger 级别 < INFO 时快速返回。
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    logger.info(f"[TIMING] {operation}: {elapsed_ms:.2f}ms")


def log_memory(logger: logging.Logger, operation: str, item_count: int):
    """记录数据量日志.

    当 logger 级别 < INFO 时快速返回。
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    logger.info(f"[DATA] {operation}: {item_count} items")


# 创建根日志记录器
logger = get_logger("md2rag")
