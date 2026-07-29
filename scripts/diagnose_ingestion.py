#!/usr/bin/env python3
"""
数据入库诊断脚本 - 用于排查入库中途退出的问题

用法:
    python diagnose_ingestion.py [classification]

示例:
    python diagnose_ingestion.py public
    python diagnose_ingestion.py confidential
    python diagnose_ingestion.py restricted
    python diagnose_ingestion.py all
"""

import sys
import os
import time
import traceback
import signal
from pathlib import Path
from datetime import datetime

# 尝试导入 psutil
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[INFO] psutil 未安装，进程监控功能受限。安装命令: pip install psutil")

# 添加 backend 到路径
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.services.ingestion import data_ingestion_service


class ProcessMonitor:
    """监控进程资源使用情况."""

    def __init__(self, interval=5):
        self.interval = interval
        self.running = False
        self.start_time = time.time()
        self.peak_memory = 0

        if HAS_PSUTIL:
            self.process = psutil.Process()
        else:
            self.process = None

    def start(self):
        """开始监控."""
        self.running = True
        import threading
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        pid = self.process.pid if self.process else os.getpid()
        print(f"[MONITOR] 进程监控已启动 (PID: {pid})")

    def _monitor_loop(self):
        """监控循环."""
        while self.running:
            try:
                if HAS_PSUTIL and self.process:
                    mem_info = self.process.memory_info()
                    cpu_percent = self.process.cpu_percent(interval=1)
                    mem_mb = mem_info.rss / 1024 / 1024
                    self.peak_memory = max(self.peak_memory, mem_mb)
                    num_threads = self.process.num_threads()
                else:
                    mem_mb = 0
                    cpu_percent = 0
                    num_threads = 0

                elapsed = time.time() - self.start_time
                print(f"\n[MONITOR] 运行时间: {elapsed:.1f}s | "
                      f"CPU: {cpu_percent:.1f}% | "
                      f"内存: {mem_mb:.1f}MB (峰值: {self.peak_memory:.1f}MB) | "
                      f"线程: {num_threads}")
            except Exception as e:
                print(f"[MONITOR] 监控错误: {e}")
            time.sleep(self.interval)

    def stop(self):
        """停止监控."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        print(f"[MONITOR] 监控已停止，峰值内存: {self.peak_memory:.1f}MB")


def signal_handler(signum, frame):
    """信号处理."""
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    print(f"\n[SIGNAL] 收到信号 {sig_name}")
    print(f"[SIGNAL] 堆栈跟踪:\n{''.join(traceback.format_stack(frame))}")
    sys.exit(128 + signum)


def main():
    # 注册信号处理
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, signal_handler)
        except (OSError, ValueError):
            pass

    # 解析参数
    classification = sys.argv[1] if len(sys.argv) > 1 else "all"
    levels = ["public", "confidential", "restricted"] if classification == "all" else [classification]

    print("=" * 80)
    print("🔍 数据入库诊断工具")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"PID: {os.getpid()}")
    print(f"目标密级: {levels}")
    print("=" * 80)

    # 启动监控
    monitor = ProcessMonitor(interval=10)
    monitor.start()

    try:
        for level in levels:
            print(f"\n{'='*80}")
            print(f"📂 开始导入 {level.upper()} 级别文档")
            print(f"{'='*80}")

            level_start = time.time()

            try:
                # 预估
                print(f"\n[STEP] 预估 {level} 记录数...")
                estimate = data_ingestion_service.estimate_records(level)
                print(f"[INFO] 预估结果: {estimate}")

                if estimate.get("file_count", 0) == 0:
                    print(f"[SKIP] {level} 无文件，跳过")
                    continue

                # 执行导入
                print(f"\n[STEP] 开始导入 {level}...")
                result = data_ingestion_service.ingest_directory(level)
                print(f"\n[RESULT] {level} 导入结果:")
                for key, value in result.items():
                    print(f"  {key}: {value}")

            except Exception as e:
                print(f"\n[ERROR] {level} 导入失败!")
                print(f"[ERROR] 异常类型: {type(e).__name__}")
                print(f"[ERROR] 异常信息: {e}")
                print(f"[ERROR] 堆栈跟踪:")
                traceback.print_exc()

            level_elapsed = time.time() - level_start
            print(f"[TIME] {level} 耗时: {level_elapsed:.1f}s")

    except Exception as e:
        print(f"\n[CRITICAL] 致命错误!")
        print(f"[CRITICAL] 异常类型: {type(e).__name__}")
        print(f"[CRITICAL] 异常信息: {e}")
        traceback.print_exc()

    finally:
        monitor.stop()
        print(f"\n{'='*80}")
        print("诊断完成")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总运行时间: {time.time() - monitor.start_time:.1f}s")
        print(f"峰值内存: {monitor.peak_memory:.1f}MB")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()