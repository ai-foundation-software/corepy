#!/usr/bin/env python3
"""
benchmarks/cpu_matmul_bench.py
-------------------------------
Compare CorePy vs NumPy matrix multiplication across:
  - Multiple matrix sizes (64 → 4096)
  - Multiple backend policies (auto, openblas, rust)
  - Reports: median time, GFLOP/s, speedup vs NumPy

Usage:
    python benchmarks/cpu_matmul_bench.py
    python benchmarks/cpu_matmul_bench.py --sizes 256 512 1024 2048 4096
    python benchmarks/cpu_matmul_bench.py --warmup 3 --reps 10
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Optional

# Reconfigure stdout to UTF-8 on Windows cp1252/non-UTF-8 terminals if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

# ─── Optional: load CorePy ───────────────────────────────────────────────────
try:
    import corepy.matmul as cm

    COREPY_AVAILABLE = True
except ImportError as e:
    cm = None  # type: ignore
    COREPY_AVAILABLE = False
    print(f"[warn] CorePy not importable: {e}")
    print("       Only NumPy will be benchmarked.")
    print(
        "       Run: .venv/bin/maturin develop --release --manifest-path rust/core/Cargo.toml"
    )


# ============================================================================
# Benchmark Helpers
# ============================================================================


def gflops(m: int, n: int, k: int, elapsed_sec: float) -> float:
    """Return GFLOP/s for a (m,n,k) matmul completed in `elapsed_sec` seconds."""
    return 2.0 * m * n * k / elapsed_sec / 1e9


def bench(fn, warmup: int = 3, reps: int = 10) -> float:
    """Run `fn()` `warmup+reps` times; return median wall-clock time in seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def make_matrices(size: int, dtype=np.float32):
    rng = np.random.default_rng(42)
    a = rng.standard_normal((size, size)).astype(dtype, copy=False)
    b = rng.standard_normal((size, size)).astype(dtype, copy=False)
    return a, b


# ============================================================================
# Benchmark Runs
# ============================================================================


def run_numpy(a, b, warmup: int, reps: int):
    return bench(lambda: np.matmul(a, b), warmup, reps)


def run_corepy(a, b, policy: str, warmup: int, reps: int):
    out = np.empty((a.shape[0], b.shape[1]), dtype=np.float32)
    return bench(lambda: cm.matmul(a, b, policy=policy, out=out), warmup, reps)


def format_row(label, t_sec, gflop_s, speedup=None):
    sp = f"  {speedup:6.2f}x" if speedup is not None else "   --   "
    return f"  {label:<20s}  {t_sec * 1000:8.2f} ms  {gflop_s:7.2f} GF/s  {sp}"


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="CorePy vs NumPy matmul benchmark")
    parser.add_argument(
        "--sizes",
        "--size",
        nargs="+",
        type=int,
        default=[64, 128, 256, 512, 1024, 2048, 4096],
        help="Matrix sizes (NxN) to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--reps", type=int, default=10, help="Benchmark repetitions")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["auto", "openblas", "rust"],
        help="CorePy backend policies to benchmark",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  CorePy CPU Matmul Benchmark  (float32, NxN @ NxN)")
    print("=" * 70)

    if COREPY_AVAILABLE:
        info = cm.backend_info()
        print(f"\n  CPU     : {info['brand']}")
        print(f"  Vendor  : {info['vendor']}")
        print(f"  Backend : {info['backend']}")
        print(
            f"  Cores   : {info['physical_cores']} physical / {info['logical_cores']} logical"
        )
        print(f"  Threads : {info['threads']} (for 2048x2048 matmul)")
    print()

    for size in args.sizes:
        a, b = make_matrices(size)
        m = n = k = size

        print(
            f"  --- N = {size:4d} ({'%.1f' % (2 * m * n * k / 1e9)} GFLOPs per call) ---"
        )

        # NumPy baseline
        t_np = run_numpy(a, b, args.warmup, args.reps)
        np_gf = gflops(m, n, k, t_np)
        print(format_row("numpy", t_np, np_gf))

        if not COREPY_AVAILABLE:
            print()
            continue

        # CorePy with each policy
        for policy in args.policies:
            try:
                t_cp = run_corepy(a, b, policy, args.warmup, args.reps)
                cp_gf = gflops(m, n, k, t_cp)
                speedup = t_np / t_cp
                print(format_row(f"corepy [{policy}]", t_cp, cp_gf, speedup))
            except Exception as e:
                print(f"  corepy [{policy}]  ERROR: {e}")

        print()

    print("  Legend:  ms=milliseconds  GF/s=GFLOP/s  speedup vs numpy")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
