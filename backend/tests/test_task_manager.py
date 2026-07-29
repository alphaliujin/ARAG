"""TaskManager 单元测试 — 重点验证之前发现的 race conditions:
1. start_task 不应覆盖 CANCELLED 状态
2. cleanup_old_tasks size cap 正确裁剪
3. _worker 启动后取消能被识别
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

# 让测试不依赖 backend 在 sys.path
_HERE = Path(__file__).resolve().parent
_BACKEND = _HERE.parent.parent / "backend"
sys.path.insert(0, str(_BACKEND))

from app.services.task_manager import (  # noqa: E402
    TaskManager,
    TASK_PENDING,
    TASK_RUNNING,
    TASK_COMPLETED,
    TASK_CANCELLED,
)


class TestTaskManagerCancellationRace(unittest.TestCase):
    def test_start_task_does_not_overwrite_cancelled(self):
        """在 worker 执行 start_task 之前用户已 cancel: start_task 必须不动."""
        tm = TaskManager()
        tid = tm.create_task("preprocess")

        # 模拟用户先取消(在 worker 还没跑 start_task 前)
        ok = tm.cancel_task(tid)
        self.assertTrue(ok)
        self.assertEqual(tm.get_task(tid).status, TASK_CANCELLED)

        # 此时 worker 调用 start_task — 不应翻回 RUNNING
        tm.start_task(tid)
        self.assertEqual(tm.get_task(tid).status, TASK_CANCELLED,
                         "start_task 不应覆盖已 CANCELLED 的任务")

    def test_run_in_thread_respects_cancel_before_start(self):
        """create→cancel→worker 序列: 任务最终为 CANCELLED, func 不被调用."""
        tm = TaskManager()
        called = {"n": 0}

        def slow_func():
            called["n"] += 1
            time.sleep(0.5)
            return {"ok": True}

        tid = tm.create_task("preprocess")
        # 先取消
        tm.cancel_task(tid)
        # 再让 worker 跑
        tm.run_in_thread(tid, slow_func)
        time.sleep(0.3)

        info = tm.get_task(tid)
        self.assertEqual(info.status, TASK_CANCELLED)
        # func 不应被调用 (worker 在 start_task 后立刻检测到 status!=RUNNING 退出)
        self.assertEqual(called["n"], 0)


class TestTaskManagerCleanup(unittest.TestCase):
    def test_size_cap_drops_oldest_terminal_tasks(self):
        """size cap 触发时,留最新的 N 个."""
        tm = TaskManager()
        now = time.time()
        # 制造 10 个完成任务,带不同 completed_at (都在最近 1 分钟内,不会被 24h 时间过期裁掉)
        for i in range(10):
            tid = tm.create_task("dedup")
            tm.start_task(tid)
            tm.complete_task(tid, {"i": i})
            with tm._lock:
                tm._tasks[tid].completed_at = now - 60 + i  # 比当前早 60s,但都比 24h 新

        # max_completed=3
        tm.cleanup_old_tasks(max_age_seconds=86400, max_completed=3)
        with tm._lock:
            remaining = list(tm._tasks.values())
        self.assertEqual(len(remaining), 3)
        # 保留最新的 3 个 (i=7,8,9 对应 completed_at = now-53, now-52, now-51)
        kept_completed = sorted(t.completed_at for t in remaining)
        expected = sorted([now - 60 + i for i in (7, 8, 9)])
        for a, b in zip(kept_completed, expected):
            self.assertAlmostEqual(a, b, places=3)

    def test_size_cap_preserves_running(self):
        """size cap 只裁终态任务,running/pending 不受影响."""
        tm = TaskManager()
        now = time.time()
        # 5 个 completed
        for i in range(5):
            tid = tm.create_task("ingest")
            tm.start_task(tid)
            tm.complete_task(tid)
            with tm._lock:
                tm._tasks[tid].completed_at = now - 60 + i
        # 1 个 running
        running_tid = tm.create_task("preprocess")
        tm.start_task(running_tid)

        tm.cleanup_old_tasks(max_age_seconds=86400, max_completed=2)
        with tm._lock:
            statuses = sorted(t.status for t in tm._tasks.values())
        # 应有 2 completed + 1 running
        self.assertEqual(statuses.count(TASK_COMPLETED), 2)
        self.assertEqual(statuses.count(TASK_RUNNING), 1)


class TestTaskManagerUpdateProgressGuards(unittest.TestCase):
    def test_update_progress_ignored_after_completion(self):
        """已完成任务的 progress 更新应被忽略,防止迟到的 callback 覆盖结果."""
        tm = TaskManager()
        tid = tm.create_task("preprocess")
        tm.start_task(tid)
        tm.complete_task(tid, {"a": 1})

        tm.update_progress(tid, 0.5, {"message": "late"})
        info = tm.get_task(tid)
        self.assertEqual(info.progress, 1.0)
        self.assertEqual(info.status, TASK_COMPLETED)


if __name__ == "__main__":
    unittest.main()
