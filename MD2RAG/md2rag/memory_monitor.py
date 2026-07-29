"""内存监控工具 - 用于调试入库过程中的内存问题.

使用方法:
    from md2rag.memory_monitor import MemoryMonitor

    monitor = MemoryMonitor()
    monitor.start()

    # ... 执行入库操作 ...

    monitor.stop()
    monitor.report()
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# 跨平台内存获取
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class MemorySnapshot:
    """内存快照."""
    timestamp: float
    rss_mb: float          # 进程驻留内存 (MB)
    vms_mb: float          # 进程虚拟内存 (MB)
    available_mb: float    # 系统可用内存 (MB)
    total_mb: float        # 系统总内存 (MB)
    percent: float         # 进程内存占用百分比
    torch_cuda_mb: float = 0.0   # CUDA 显存 (MB)
    torch_mps_mb: float = 0.0    # MPS 显存 (MB)
    label: str = ""        # 快照标签
    stack_hint: str = ""   # 调用栈提示


class MemoryMonitor:
    """内存监控器.

    功能:
    1. 定时采样内存使用
    2. 检测内存泄漏（持续增长）
    3. 生成内存报告
    4. 支持阈值告警
    """

    def __init__(
        self,
        sample_interval: float = 1.0,      # 采样间隔（秒）
        leak_threshold_mb: float = 100.0,   # 泄漏阈值（连续增长 MB）
        leak_samples: int = 5,              # 泄漏检测采样数
        warning_percent: float = 80.0,      # 内存占用告警阈值
        callback: Optional[Callable[[str, Dict], None]] = None,  # 告警回调
    ):
        self.sample_interval = sample_interval
        self.leak_threshold_mb = leak_threshold_mb
        self.leak_samples = leak_samples
        self.warning_percent = warning_percent
        self.callback = callback

        self._snapshots: List[MemorySnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

        if not HAS_PSUTIL:
            print("[MEMORY_MONITOR] psutil not installed, memory tracking will be limited")
            print("[MEMORY_MONITOR] Install with: pip install psutil")

    def _get_memory_info(self) -> Dict[str, float]:
        """获取当前内存信息."""
        info = {
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "available_mb": 0.0,
            "total_mb": 0.0,
            "percent": 0.0,
        }

        if HAS_PSUTIL:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            vm_info = psutil.virtual_memory()

            info["rss_mb"] = mem_info.rss / 1024 / 1024
            info["vms_mb"] = mem_info.vms / 1024 / 1024
            info["available_mb"] = vm_info.available / 1024 / 1024
            info["total_mb"] = vm_info.total / 1024 / 1024
            info["percent"] = process.memory_percent()

        # PyTorch 显存
        if HAS_TORCH:
            if torch.cuda.is_available():
                info["torch_cuda_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    # MPS 没有直接的内存查询 API，用系统内存近似
                    info["torch_mps_mb"] = info["rss_mb"] * 0.3  # 粗略估计
                except Exception:
                    pass

        return info

    def _take_snapshot(self, label: str = "") -> MemorySnapshot:
        """拍摄内存快照."""
        info = self._get_memory_info()

        # 获取调用栈提示（最近几层）
        stack = traceback.extract_stack(limit=5)
        stack_hint = " -> ".join(f"{f.name}:{f.lineno}" for f in stack[-3:])

        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=info["rss_mb"],
            vms_mb=info["vms_mb"],
            available_mb=info["available_mb"],
            total_mb=info["total_mb"],
            percent=info["percent"],
            torch_cuda_mb=info.get("torch_cuda_mb", 0.0),
            torch_mps_mb=info.get("torch_mps_mb", 0.0),
            label=label,
            stack_hint=stack_hint,
        )

    def _monitor_loop(self):
        """监控循环."""
        while self._running:
            snapshot = self._take_snapshot()
            self._snapshots.append(snapshot)

            # 检测内存告警
            if snapshot.percent > self.warning_percent:
                self._alert("warning", {
                    "message": f"Memory usage {snapshot.percent:.1f}% exceeds threshold {self.warning_percent}%",
                    "rss_mb": snapshot.rss_mb,
                    "percent": snapshot.percent,
                })

            # 检测内存泄漏
            if len(self._snapshots) >= self.leak_samples:
                recent = self._snapshots[-self.leak_samples:]
                deltas = [recent[i+1].rss_mb - recent[i].rss_mb for i in range(len(recent)-1)]
                if all(d > 0 for d in deltas) and sum(deltas) > self.leak_threshold_mb:
                    self._alert("leak", {
                        "message": f"Potential memory leak detected: RSS grew by {sum(deltas):.1f}MB over {self.leak_samples} samples",
                        "deltas": deltas,
                        "current_rss_mb": snapshot.rss_mb,
                    })

            time.sleep(self.sample_interval)

    def _alert(self, alert_type: str, data: Dict):
        """发送告警."""
        print(f"[MEMORY_ALERT][{alert_type.upper()}] {data.get('message', '')}")
        if self.callback:
            try:
                self.callback(alert_type, data)
            except Exception as e:
                print(f"[MEMORY_MONITOR] Alert callback failed: {e}")

    def start(self):
        """启动监控."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._snapshots = []

        # 记录起始快照
        self._snapshots.append(self._take_snapshot("start"))

        # 启动监控线程
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print(f"[MEMORY_MONITOR] Started (interval={self.sample_interval}s)")

    def stop(self):
        """停止监控."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        # 记录结束快照
        self._snapshots.append(self._take_snapshot("stop"))
        print("[MEMORY_MONITOR] Stopped")

    def checkpoint(self, label: str = ""):
        """记录检查点快照."""
        snapshot = self._take_snapshot(label)
        self._snapshots.append(snapshot)
        print(f"[MEMORY_MONITOR] Checkpoint '{label}': RSS={snapshot.rss_mb:.1f}MB, Available={snapshot.available_mb:.1f}MB")

    def report(self) -> Dict[str, Any]:
        """生成内存报告."""
        if not self._snapshots:
            return {"error": "No snapshots recorded"}

        start = self._snapshots[0]
        end = self._snapshots[-1]
        peak = max(self._snapshots, key=lambda s: s.rss_mb)

        # 内存增长趋势
        deltas = []
        for i in range(1, len(self._snapshots)):
            prev = self._snapshots[i-1]
            curr = self._snapshots[i]
            deltas.append({
                "from": prev.label or f"t{i-1}",
                "to": curr.label or f"t{i}",
                "delta_mb": curr.rss_mb - prev.rss_mb,
                "elapsed_s": curr.timestamp - prev.timestamp,
            })

        # 统计
        report = {
            "duration_s": end.timestamp - start.timestamp,
            "start_rss_mb": start.rss_mb,
            "end_rss_mb": end.rss_mb,
            "peak_rss_mb": peak.rss_mb,
            "delta_rss_mb": end.rss_mb - start.rss_mb,
            "total_snapshots": len(self._snapshots),
            "sample_interval_s": self.sample_interval,
            "deltas": deltas,
            "snapshots": [
                {
                    "time": s.timestamp - start.timestamp,
                    "rss_mb": s.rss_mb,
                    "percent": s.percent,
                    "label": s.label,
                }
                for s in self._snapshots
            ],
        }

        # 打印摘要
        print("\n" + "=" * 60)
        print("MEMORY REPORT")
        print("=" * 60)
        print(f"Duration: {report['duration_s']:.1f}s")
        print(f"Start RSS: {start.rss_mb:.1f} MB")
        print(f"End RSS: {end.rss_mb:.1f} MB")
        print(f"Peak RSS: {peak.rss_mb:.1f} MB")
        print(f"Delta: {report['delta_rss_mb']:+.1f} MB")
        print("=" * 60)

        # 打印显著增长点
        significant_deltas = [d for d in deltas if abs(d["delta_mb"]) > 10]
        if significant_deltas:
            print("\nSignificant memory changes (>10MB):")
            for d in significant_deltas:
                print(f"  {d['from']} -> {d['to']}: {d['delta_mb']:+.1f} MB in {d['elapsed_s']:.1f}s")

        return report

    def force_gc(self):
        """强制垃圾回收并记录效果."""
        before = self._get_memory_info()["rss_mb"]
        gc.collect()
        if HAS_TORCH:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
        after = self._get_memory_info()["rss_mb"]
        freed = before - after
        print(f"[MEMORY_MONITOR] GC freed {freed:.1f} MB (before={before:.1f}, after={after:.1f})")
        return freed


# 全局监控器实例（方便使用）
_global_monitor: Optional[MemoryMonitor] = None


def get_monitor() -> MemoryMonitor:
    """获取全局监控器."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor


def start_monitoring(**kwargs):
    """启动全局监控."""
    get_monitor().start()


def stop_monitoring():
    """停止全局监控."""
    get_monitor().stop()


def memory_checkpoint(label: str = ""):
    """记录内存检查点."""
    get_monitor().checkpoint(label)


def memory_report() -> Dict[str, Any]:
    """生成内存报告."""
    return get_monitor().report()


def force_gc() -> float:
    """强制垃圾回收."""
    return get_monitor().force_gc()
