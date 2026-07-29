"""OllamaEmbedder 单元测试 — 用 mock HTTP server 覆盖 retry/batch/cache/dimension 路径,
不依赖真实 Ollama,可在 CI 离线运行。"""
from __future__ import annotations

import json
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from md2rag.embedder import OllamaEmbedder


class _MockOllamaHandler(BaseHTTPRequestHandler):
    """每次 POST 调用 server.responder(payload) 决定返回 status + body."""

    def log_message(self, *args, **kwargs):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}
        responder: Callable[[dict, str], tuple[int, dict]] = self.server.responder  # type: ignore[attr-defined]
        status, resp = responder(payload, self.path)
        encoded = json.dumps(resp).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _run_mock(responder):
    server = HTTPServer(("127.0.0.1", 0), _MockOllamaHandler)
    server.responder = responder  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


class TestOllamaEmbedderHappyPath(unittest.TestCase):
    """正常路径: 单条/批量都返回归一化向量."""

    def test_single_text_returns_normalized_vector(self):
        def responder(payload, path):
            return 200, {"embedding": [3.0, 4.0]}  # |v|=5

        server, base = _run_mock(responder)
        try:
            embedder = OllamaEmbedder(base_url=base, model="test:latest", max_retries=1, retry_delay=0.0)
            result = embedder.embed(["hello"])
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]), 2)
            # 应该已归一化: [0.6, 0.8]
            self.assertAlmostEqual(result[0][0], 0.6, places=4)
            self.assertAlmostEqual(result[0][1], 0.8, places=4)
        finally:
            server.shutdown()

    def test_batch_path_uses_api_embed(self):
        captured_paths = []

        def responder(payload, path):
            captured_paths.append(path)
            if path == "/api/embed":
                texts = payload.get("input", [])
                return 200, {"embeddings": [[1.0, 0.0] for _ in texts]}
            return 200, {"embedding": [1.0, 0.0]}

        server, base = _run_mock(responder)
        try:
            embedder = OllamaEmbedder(base_url=base, model="test:latest", max_retries=1, retry_delay=0.0)
            result = embedder.embed(["a", "b", "c"])
            self.assertEqual(len(result), 3)
            # 批量路径走 /api/embed
            self.assertIn("/api/embed", captured_paths)
        finally:
            server.shutdown()

    def test_cache_avoids_repeat_calls(self):
        call_count = {"n": 0}

        def responder(payload, path):
            call_count["n"] += 1
            return 200, {"embedding": [1.0, 0.0]}

        server, base = _run_mock(responder)
        try:
            embedder = OllamaEmbedder(base_url=base, model="test:latest", max_retries=1, retry_delay=0.0)
            embedder.embed(["alpha"])
            cnt1 = call_count["n"]
            embedder.embed(["alpha"])
            self.assertEqual(call_count["n"], cnt1, "second call should hit cache, not Ollama")
        finally:
            server.shutdown()


class TestOllamaEmbedderRetryAndError(unittest.TestCase):
    """失败路径: 临时错误重试 + 永久失败抛 RuntimeError."""

    def test_retries_on_500_then_succeeds(self):
        attempts = {"n": 0}

        def responder(payload, path):
            attempts["n"] += 1
            if attempts["n"] < 3:
                return 500, {"error": "transient"}
            return 200, {"embedding": [1.0, 0.0]}

        server, base = _run_mock(responder)
        try:
            embedder = OllamaEmbedder(base_url=base, model="t:l", max_retries=3, retry_delay=0.01)
            out = embedder.embed(["x"])
            self.assertEqual(len(out), 1)
            self.assertEqual(attempts["n"], 3)
        finally:
            server.shutdown()

    def test_all_retries_failing_raises(self):
        def responder(payload, path):
            return 500, {"error": "permanent"}

        server, base = _run_mock(responder)
        try:
            embedder = OllamaEmbedder(base_url=base, model="t:l", max_retries=2, retry_delay=0.01)
            with self.assertRaises(RuntimeError):
                embedder.embed(["x"])
        finally:
            server.shutdown()


class TestOllamaEmbedderDimension(unittest.TestCase):
    """dimension 探测: 成功 → 实际长度;失败 → 抛 RuntimeError (不再猜 1024)."""

    def test_dimension_probe_success(self):
        def responder(payload, path):
            return 200, {"embedding": [0.1] * 128}

        server, base = _run_mock(responder)
        try:
            embedder = OllamaEmbedder(base_url=base, model="t:l", max_retries=1, retry_delay=0.0)
            self.assertEqual(embedder.dimension, 128)
        finally:
            server.shutdown()

    def test_dimension_probe_empty_raises(self):
        def responder(payload, path):
            return 200, {"embedding": []}

        server, base = _run_mock(responder)
        try:
            embedder = OllamaEmbedder(base_url=base, model="t:l", max_retries=1, retry_delay=0.0)
            with self.assertRaises(RuntimeError):
                _ = embedder.dimension
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
