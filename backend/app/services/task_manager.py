"""后台任务管理器 — 在进程内维护任务注册表,支持后台线程执行 + 实时进度查询。

核心场景: 预处理/入库/去重等长时间操作开始后,用户关闭浏览器或切换页面,
后台线程继续运行;再次打开页面时,通过 GET /tasks/{id} 查看进度。

架构选择:
- 不用 Celery/Redis(太重),纯 Python threading + dict 注册表
- 不用 SSE(浏览器关了连接断),用简单轮询(GET /tasks/{id})
- Daemon 线程: uvicorn 进程退出时自动终止,开发期可接受
- 生产期: 去掉 --reload 即稳定持久
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_COMPLETED = "completed"
TASK_FAILED = "failed"
TASK_CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """单个后台任务的状态记录."""
    task_id: str
    task_type: str          # "preprocess" | "ingest" | "dedup"
    status: str = TASK_PENDING
    progress: float = 0.0   # 0.0 ~ 1.0
    progress_detail: dict = field(default_factory=dict)
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    params: dict = field(default_factory=dict)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancel_event(self) -> threading.Event:
        """工作线程可检查此事件以实现协作式取消."""
        return self._cancel_event

    def is_cancelled(self) -> bool:
        """快捷方法: 检查取消标志是否已设置."""
        return self._cancel_event.is_set()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress": self.progress,
            "progress_detail": self.progress_detail,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "params": self.params,
        }


class TaskManager:
    """进程内任务注册表,线程安全."""

    def __init__(self):
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 任务生命周期
    # ------------------------------------------------------------------

    def create_task(self, task_type: str, params: dict = None) -> str:
        """创建任务记录,返回 task_id."""
        # 自动清理: 每次创建新任务时顺便清理过期任务,防止内存无限增长
        self.cleanup_old_tasks()

        task_id = str(uuid.uuid4())
        info = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            params=params or {},
        )
        with self._lock:
            self._tasks[task_id] = info
        return task_id

    def start_task(self, task_id: str):
        """标记任务开始运行 — 仅从 PENDING 状态提升,防止覆盖 CANCELLED."""
        with self._lock:
            info = self._tasks.get(task_id)
            if info and info.status == TASK_PENDING:
                info.status = TASK_RUNNING
                info.started_at = time.time()

    def update_progress(self, task_id: str, progress: float, detail: dict = None):
        """更新进度(0~1)和详情字典 — 仅对 RUNNING/PENDING 任务生效.

        已完成/失败/取消的任务拒绝进度更新,防止回调竞态覆盖终态.
        """
        with self._lock:
            info = self._tasks.get(task_id)
            if info and info.status in (TASK_PENDING, TASK_RUNNING):
                info.progress = max(0.0, min(1.0, progress))
                if detail:
                    info.progress_detail = detail

    def complete_task(self, task_id: str, result: dict = None):
        """标记任务完成 — 仅 RUNNING 状态可转 COMPLETED."""
        with self._lock:
            info = self._tasks.get(task_id)
            if info and info.status == TASK_RUNNING:
                info.status = TASK_COMPLETED
                info.progress = 1.0
                info.result = result
                info.completed_at = time.time()

    def fail_task(self, task_id: str, error: str):
        """标记任务失败 — 仅 RUNNING 状态可转 FAILED."""
        with self._lock:
            info = self._tasks.get(task_id)
            if info and info.status == TASK_RUNNING:
                info.status = TASK_FAILED
                info.error = error
                info.completed_at = time.time()

    def cancel_task(self, task_id: str) -> bool:
        """请求取消任务(协作式).

        设置取消标志,工作线程需自行检查 is_cancelled() / cancel_event
        并主动退出。返回 True 表示成功设置取消标志。
        """
        with self._lock:
            info = self._tasks.get(task_id)
            if not info or info.status not in (TASK_PENDING, TASK_RUNNING):
                return False
            info._cancel_event.set()
            info.status = TASK_CANCELLED
            info.completed_at = time.time()
        return True

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        with self._lock:
            info = self._tasks.get(task_id)
            # Return a shallow copy so the caller gets a consistent snapshot,
            # not a live mutable reference that could change between reads.
            # (progress_detail dict is still shared — callers should only read it)
            if info is None:
                return None
            return TaskInfo(
                task_id=info.task_id,
                task_type=info.task_type,
                status=info.status,
                progress=info.progress,
                progress_detail=info.progress_detail,
                result=info.result,
                error=info.error,
                created_at=info.created_at,
                started_at=info.started_at,
                completed_at=info.completed_at,
                params=info.params,
                _cancel_event=info._cancel_event,
            )

    def get_active_tasks(self) -> List[TaskInfo]:
        """返回所有 pending/running 任务(副本)."""
        with self._lock:
            return [
                TaskInfo(
                    task_id=t.task_id,
                    task_type=t.task_type,
                    status=t.status,
                    progress=t.progress,
                    progress_detail=t.progress_detail,
                    result=t.result,
                    error=t.error,
                    created_at=t.created_at,
                    started_at=t.started_at,
                    completed_at=t.completed_at,
                    params=t.params,
                    _cancel_event=t._cancel_event,
                )
                for t in self._tasks.values()
                if t.status in (TASK_PENDING, TASK_RUNNING)
            ]

    def get_task_by_type(self, task_type: str) -> Optional[TaskInfo]:
        """查找指定类型最近的 running/pending 任务(副本)."""
        with self._lock:
            candidates = [
                t for t in self._tasks.values()
                if t.task_type == task_type and t.status in (TASK_PENDING, TASK_RUNNING)
            ]
            best = max(candidates, key=lambda t: t.created_at) if candidates else None
            if best is None:
                return None
            return TaskInfo(
                task_id=best.task_id,
                task_type=best.task_type,
                status=best.status,
                progress=best.progress,
                progress_detail=best.progress_detail,
                result=best.result,
                error=best.error,
                created_at=best.created_at,
                started_at=best.started_at,
                completed_at=best.completed_at,
                params=best.params,
                _cancel_event=best._cancel_event,
            )

    def get_latest_task_by_type(self, task_type: str) -> Optional[TaskInfo]:
        """查找指定类型最近的任务(副本,不限状态)."""
        with self._lock:
            candidates = [
                t for t in self._tasks.values()
                if t.task_type == task_type
            ]
            best = max(candidates, key=lambda t: t.created_at) if candidates else None
            if best is None:
                return None
            return TaskInfo(
                task_id=best.task_id,
                task_type=best.task_type,
                status=best.status,
                progress=best.progress,
                progress_detail=best.progress_detail,
                result=best.result,
                error=best.error,
                created_at=best.created_at,
                started_at=best.started_at,
                completed_at=best.completed_at,
                params=best.params,
                _cancel_event=best._cancel_event,
            )

    def has_active_task_of_type(self, task_type: str) -> bool:
        """是否存在同类型的运行任务(用于防止重复启动)."""
        return self.get_task_by_type(task_type) is not None

    # ------------------------------------------------------------------
    # 后台执行
    # ------------------------------------------------------------------

    def run_in_thread(self, task_id: str, func: Callable, *args, **kwargs):
        """在 daemon 线程中执行 func,自动管理任务状态."""
        def _worker():
            self.start_task(task_id)
            # 启动后立即重检状态:若 create_task → cancel_task 之间已被取消,
            # start_task 会因 status != PENDING 而不提升,此处直接退出。
            with self._lock:
                info = self._tasks.get(task_id)
                if not info or info.status != TASK_RUNNING:
                    return
            try:
                result = func(*args, **kwargs)
                # Atomic check-and-complete: acquire lock, verify not cancelled, then complete
                with self._lock:
                    info = self._tasks.get(task_id)
                    if info and info.status == TASK_CANCELLED:
                        return
                    if info and info.status == TASK_RUNNING:
                        info.status = TASK_COMPLETED
                        info.progress = 1.0
                        info.result = result
                        info.completed_at = time.time()
            except Exception as e:
                # 取消导致的异常不算失败,状态已经是 cancelled
                with self._lock:
                    info = self._tasks.get(task_id)
                    if info and info.status == TASK_CANCELLED:
                        return
                    if info and info.status == TASK_RUNNING:
                        info.status = TASK_FAILED
                        info.error = str(e)
                        info.completed_at = time.time()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def make_progress_callback(self, task_id: str) -> Callable:
        """生成适配 progress_callback 签名的闭包.

        callback 接收 dict,内容示例:
          入库: {"phase": "start", "level": "public", "message": "..."}
          入库: {"phase": "indexing", "progress": 0.5, "level": "public", "message": "..."}
          入库: {"phase": "done",  "level": "restricted", "parents": 15, "children": 75}
          去重: {"phase": "progress", "progress": 0.35, "completed": 500, "total": 1500, ...}

        对于入库任务，三级密级各占 1/3 的全局进度:
          public:     0.00 ~ 0.33
          restricted: 0.33 ~ 0.67
          confidential: 0.67 ~ 1.00
        """
        # 三级入库的全局进度区间
        LEVEL_RANGES = {
            "public":      (0.00, 0.33),
            "restricted":   (0.33, 0.67),
            "confidential": (0.67, 1.00),
        }

        def _callback(detail: dict):
            phase = detail.get("phase", "")
            level = detail.get("level", "")

            # 优先使用调用方提供的精确 progress 值
            if "progress" in detail:
                # 如果是入库任务且有密级信息，将 index-level 进度映射到全局进度
                local_progress = detail["progress"]
                level_range = LEVEL_RANGES.get(level)
                if level_range:
                    base, top = level_range
                    progress = base + local_progress * (top - base)
                else:
                    # 去重等任务：直接使用提供的 progress 值
                    progress = local_progress
            elif phase == "done":
                # 三级入库: public=0.33, restricted=0.67, confidential=1.0
                level_map = {"public": 0.33, "restricted": 0.67, "confidential": 1.0}
                progress = level_map.get(level, 0.5)
            elif phase == "skipped":
                level_map = {"public": 0.33, "restricted": 0.67, "confidential": 1.0}
                progress = level_map.get(level, 0.5)
            elif phase == "start":
                level_map = {"public": 0.05, "restricted": 0.38, "confidential": 0.72}
                progress = level_map.get(level, 0.0)
            else:
                progress = 0.1

            self.update_progress(task_id, progress, detail)

        return _callback

    # ------------------------------------------------------------------
    # 清理已完成任务(可选,防内存无限增长)
    # ------------------------------------------------------------------

    def cleanup_old_tasks(self, max_age_seconds: int = 86400, max_completed: int = 200):
        """清理任务:删除超过 max_age 的已结束任务 + 已结束任务总数超 max_completed 时
        裁掉最老的,防止长时间不重启时内存无限增长。

        旧实现只按时间删,且只在 create_task 触发;如果一直不创建任务,旧记录永不清理。
        """
        cutoff = time.time() - max_age_seconds
        with self._lock:
            terminal_statuses = (TASK_COMPLETED, TASK_FAILED, TASK_CANCELLED)
            # 1) 按时间过期
            to_delete = [
                tid for tid, info in self._tasks.items()
                if info.status in terminal_statuses
                and info.completed_at and info.completed_at < cutoff
            ]
            for tid in to_delete:
                del self._tasks[tid]

            # 2) Size cap: 已结束任务超过 max_completed 时,留最新的 max_completed 条
            terminal_items = [
                (tid, info) for tid, info in self._tasks.items()
                if info.status in terminal_statuses
            ]
            if len(terminal_items) > max_completed:
                terminal_items.sort(
                    key=lambda kv: kv[1].completed_at or kv[1].created_at,
                    reverse=True,
                )
                stale = terminal_items[max_completed:]
                for tid, _ in stale:
                    del self._tasks[tid]


# 全局单例
task_manager = TaskManager()