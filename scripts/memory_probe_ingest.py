"""入库内存探针 — 在独立进程里跑入库, 每批打 RSS, 输出最终曲线.

用法:
    cd /Users/alpha/workspace/ARAG_V0.2
    python3 scripts/memory_probe_ingest.py [public|restricted|confidential|all]

输出:
    每批一行: chunks=N rss=XXX MB delta=+YYY MB
    最后给出 RSS 时间序列, 用肉眼即可看到泄漏是 step 阶跃还是单调爬升。

依赖: psutil (大概率已装), 同 backend 进程依赖。
"""
from __future__ import annotations
import os, sys, time, gc, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "MD2RAG"))
sys.path.insert(0, str(ROOT / "backend"))

# 用 backend 同一份 settings, 保证 EMBEDDING_MODEL 一致
os.environ.setdefault("PYTHONUNBUFFERED", "1")

try:
    import psutil
except ImportError:
    print("[FATAL] psutil 未安装. pip install psutil")
    sys.exit(1)


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("level", nargs="?", default="all",
                        choices=["public", "restricted", "confidential", "all"])
    parser.add_argument("--force", action="store_true",
                        help="跳过去重预检, 强制重 embed (用于复现累积问题)")
    args = parser.parse_args()

    samples: list[tuple[float, str, float]] = []  # (t, tag, rss)
    t0 = time.time()

    def stamp(tag: str):
        samples.append((time.time() - t0, tag, rss_mb()))
        prev = samples[-2][2] if len(samples) >= 2 else samples[-1][2]
        delta = samples[-1][2] - prev
        print(f"  [{samples[-1][0]:7.1f}s] {tag:50s} rss={samples[-1][2]:7.1f}MB  Δ={delta:+7.1f}MB",
              flush=True)

    stamp("00 process start")

    # 导入触发模型预加载
    from app.services.ingestion import data_ingestion_service
    stamp("01 imported ingestion_service")

    def progress_cb(data):
        phase = data.get("phase", "")
        # 只在关键 phase 打 RSS, 否则刷屏
        if phase in ("indexing", "images", "complete", "done", "start"):
            level = data.get("level", "")
            msg = data.get("message", "")[:40]
            stamp(f"phase={phase} {level} {msg}")

    if args.level == "all":
        stamp("10 ingest_all_levels start")
        res = data_ingestion_service.ingest_all_levels(
            progress_callback=progress_cb,
            force=args.force,
        )
        stamp("20 ingest_all_levels done")
    else:
        stamp(f"10 ingest_directory({args.level}) start")
        res = data_ingestion_service.ingest_directory(
            args.level,
            progress_callback=progress_cb,
            force=args.force,
        )
        stamp(f"20 ingest_directory({args.level}) done")

    # 显式再清一次, 看清理能回多少
    data_ingestion_service.cleanup_memory()
    gc.collect()
    stamp("30 after cleanup_memory")

    print()
    print("=" * 80)
    print("RSS 时间序列:")
    print("=" * 80)
    for t, tag, mb in samples:
        print(f"  {t:7.1f}s  {mb:7.1f}MB  {tag}")
    print()
    peak = max(s[2] for s in samples)
    final = samples[-1][2]
    initial = samples[0][2]
    print(f"初始 RSS: {initial:.1f} MB")
    print(f"峰值 RSS: {peak:.1f} MB  (净增长 +{peak - initial:.1f} MB)")
    print(f"末次 RSS: {final:.1f} MB  (清理后回落 {peak - final:+.1f} MB)")
    print()
    print("入库结果:", json.dumps(
        {k: v for k, v in res.items() if k != "details"},
        ensure_ascii=False, indent=2
    ))


if __name__ == "__main__":
    main()
