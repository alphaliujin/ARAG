"""持久化缓存: embedding 落盘 + 已索引文件清单 (增量索引).

- EmbeddingCache: sha256(model+text) -> vector 落盘 SQLite。进程重启后仍命中,
  避免 re-index / 集合重置后重复调用 Ollama (生成向量是入库瓶颈, ~24-115 t/s)。
- IndexManifest: 已成功索引的 chunk 文件 (path + content_hash) 清单。重跑时
  跳过未变文件, 连 ChromaDB upsert 都省掉 (add ~2300 v/s, 全量重跑仍需数分钟)。

两者均放在 vector_db_dir 下, 与向量库同级 (不进 DOC/MD 敏感原文目录)。
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from md2rag.logger import get_logger

logger = get_logger("md2rag.cache")

# SQLite 单语句参数上限远大于此, 但 IN (...) 分批更稳, 避免极端长列表报错。
_SQLITE_PARAM_LIMIT = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vec_to_bytes(vec) -> bytes:
    """list[float] -> float32 little-endian bytes (1024 dim = 4KB)."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def _bytes_to_vec(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()


class EmbeddingCache:
    """SQLite 持久化 embedding 缓存。线程安全 (单连接 + Lock, WAL 模式)。

    key = sha256(model + text), 已含 model 名 -> 换模型自动 miss, 无需手动失效。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # check_same_thread=False: embed 可能在 uvicorn 多线程被调; 用 self._lock 串行化访问。
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                 key TEXT PRIMARY KEY,
                 model TEXT NOT NULL,
                 dim INTEGER NOT NULL,
                 vec BLOB NOT NULL,
                 ts TEXT NOT NULL
               )"""
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model)")
        self._conn.commit()
        logger.info(f"[EMB_CACHE] opened {self.path}")

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        """批量查。返回 {key: vec} 仅含命中的。"""
        if not keys:
            return {}
        out: dict[str, list[float]] = {}
        with self._lock:
            for i in range(0, len(keys), _SQLITE_PARAM_LIMIT):
                chunk = keys[i:i + _SQLITE_PARAM_LIMIT]
                placeholders = ",".join("?" * len(chunk))
                cur = self._conn.execute(
                    f"SELECT key, vec FROM embeddings WHERE key IN ({placeholders})", chunk
                )
                for k, blob in cur.fetchall():
                    out[k] = _bytes_to_vec(blob)
        return out

    def put_many(self, items: list[tuple[str, str, int, list[float]]]) -> None:
        """批量写。items: [(key, model, dim, vec), ...]。"""
        if not items:
            return
        rows = [(k, m, d, _vec_to_bytes(v), _now()) for k, m, d, v in items]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, model, dim, vec, ts) VALUES (?,?,?,?,?)",
                rows,
            )
            self._conn.commit()

    def count(self, model: Optional[str] = None) -> int:
        with self._lock:
            if model:
                cur = self._conn.execute("SELECT COUNT(*) FROM embeddings WHERE model=?", (model,))
            else:
                cur = self._conn.execute("SELECT COUNT(*) FROM embeddings")
            return cur.fetchone()[0]

    def clear(self, model: Optional[str] = None) -> None:
        """清缓存。换模型时无需调 (key 含 model 自动隔离); 仅在显式清库时用。"""
        with self._lock:
            if model:
                self._conn.execute("DELETE FROM embeddings WHERE model=?", (model,))
            else:
                self._conn.execute("DELETE FROM embeddings")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass


class IndexManifest:
    """已成功索引文件清单 (增量跳过)。path -> content_hash。

    仅记录"已成功 embed+upsert"的文件; 处理中途异常的不记录 -> 下次重试。
    清库 (collection 被删) 时必须 clear, 否则 manifest 说"已索引"但向量已不存在。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS indexed_files (
                 path TEXT PRIMARY KEY,
                 content_hash TEXT NOT NULL,
                 classification TEXT,
                 ts TEXT NOT NULL
               )"""
        )
        self._conn.commit()
        logger.info(f"[MANIFEST] opened {self.path}")

    @staticmethod
    def file_hash(path: str | Path) -> str:
        """文件内容 sha256 (流式读, chunk 文件是 KB 级 JSON, 很快)。"""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 16), b""):
                h.update(block)
        return h.hexdigest()

    def is_done(self, path: str, content_hash: str) -> bool:
        """path 已记录 且 content_hash 一致 -> 未变, 可跳过。"""
        with self._lock:
            cur = self._conn.execute(
                "SELECT content_hash FROM indexed_files WHERE path=?", (str(path),)
            )
            row = cur.fetchone()
        return row is not None and row[0] == content_hash

    def mark_done(self, path: str, content_hash: str, classification: Optional[str] = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO indexed_files (path, content_hash, classification, ts) VALUES (?,?,?,?)",
                (str(path), content_hash, classification, _now()),
            )
            self._conn.commit()

    def clear(self, classification: Optional[str] = None) -> int:
        """清清单。classification=None 清全部 (配合清库)。返回删除行数。"""
        with self._lock:
            if classification:
                cur = self._conn.execute(
                    "DELETE FROM indexed_files WHERE classification=?", (classification,)
                )
            else:
                cur = self._conn.execute("DELETE FROM indexed_files")
            self._conn.commit()
            return cur.rowcount

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
