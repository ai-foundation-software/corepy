#!/usr/bin/env python3
"""
benchmarks/thread_scaling.py
-----------------------------
Measure how matmul throughput scales with thread count for the
'rust' and 'openblas' backends at a fixed matrix size.

Controls RAYON_NUM_THREADS / OPENBLAS_NUM_THREADS via environment
variables rather than the backend policy API, so it works correctly
even when the global pool has already been initialised.

Usage:
    COREPY_BACKEND=rust  python benchmarks/thread_scaling.py --size 2048
    COREPY_BACKEND=openblas python benchmarks/thread_scaling.py --size 2048
    python benchmarks/thread_scaling.py --size 1024 --policy rust
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time

import numpy as np


def gflops(n: int, t: float) -> float:
    return 2.0 * n**3 / t / 1e9


def bench_subprocess(size: int, threads: int, policy: str, reps: int) -> float | None:
    """
    Benchmark in a fresh subprocess so we can set env vars before any
    Rayon / OpenBLAS thread pool initialisation.
    """
    env = os.environ.copy()
    env["COREPY_BACKEND"] = policy

    # Propagate VIRTUAL_ENV so MKL / AOCL probing finds venv-installed libraries
    if "VIRTUAL_ENV" not in env:
        venv_dir = os.path.dirname(os.path.dirname(sys.executable))
        if os.path.isfile(os.path.join(venv_dir, "pyvenv.cfg")):
            env["VIRTUAL_ENV"] = venv_dir

    if policy in ("rust", "rustparallel"):
        env["RAYON_NUM_THREADS"] = str(threads)
    elif policy == "openblas":
        env["OPENBLAS_NUM_THREADS"] = str(threads)
    elif policy == "mkl":
        env["MKL_NUM_THREADS"] = str(threads)
    elif policy in ("aocl", "blis"):
        env["BLIS_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)

    code = f"""
import numpy as np, time, statistics, sys
try:
    import corepy.matmul as cm
except ImportError as e:
    print(f"ImportError: {{e}}", file=sys.stderr)
    sys.exit(1)

n = {size}
rng = np.random.default_rng(42)
a = rng.standard_normal((n,n)).astype(np.float32)
b = rng.standard_normal((n,n)).astype(np.float32)

for _ in range(2):  # warmup
    cm.matmul(a, b)

ts = []
for _ in range({reps}):
    t0 = time.perf_counter()
    cm.matmul(a, b)
    ts.append(time.perf_counter() - t0)

print(statistics.median(ts))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        if result.returncode != 0:
            return None, result.stderr.strip()
        return float(result.stdout.strip()), None
    except (subprocess.TimeoutExpired, ValueError) as e:
        return None, str(e)


def numpy_baseline(size: int, reps: int) -> float:
    rng = np.random.default_rng(42)
    a = rng.standard_normal((size, size)).astype(np.float32)
    b = rng.standard_normal((size, size)).astype(np.float32)
    for _ in range(2):
        np.matmul(a, b)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        np.matmul(a, b)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(
        description="Thread scaling benchmark for corepy matmul"
    )
    parser.add_argument("--size", type=int, default=2048, help="Matrix side length")
    parser.add_argument(
        "--policy",
        default="rust",
        choices=["rust", "openblas", "mkl", "aocl"],
        help="Backend policy to benchmark",
    )
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=None,
        help="Thread counts to test (default: 1 → num_cpus in powers of 2)",
    )
    parser.add_argument(
        "--reps", type=int, default=5, help="Repetitions per thread count"
    )
    args = parser.parse_args()

    import multiprocessing

    max_threads = multiprocessing.cpu_count()

    if args.threads is None:
        t = 1
        thread_list = []
        while t <= max_threads:
            thread_list.append(t)
            t *= 2
        if thread_list[-1] != max_threads:
            thread_list.append(max_threads)
    else:
        thread_list = sorted(set(args.threads))

    n = args.size
    print()
    print(f"Thread Scaling Benchmark  |  N={n}  |  policy={args.policy}")
    print("-" * 60)
    print(
        f"{'Threads':>8}  {'Median ms':>12}  {'GFLOP/s':>10}  {'Speedup vs 1T':>14}  {'vs NumPy':>10}"
    )
    print("-" * 60)

    # NumPy reference
    t_np = numpy_baseline(n, args.reps)
    gf_np = gflops(n, t_np)
    print(
        f"{'numpy':>8}  {t_np * 1000:>12.2f}  {gf_np:>10.2f}  {'—':>14}  {'1.00x':>10}"
    )

    single_thread_time = None
    for threads in thread_list:
        t, err = bench_subprocess(n, threads, args.policy, args.reps)
        if t is None:
            err_short = (err or "unknown error")[:60]
            print(f"{threads:>8}  {'ERROR':>12}  ({err_short})")
            continue
        gf = gflops(n, t)
        if single_thread_time is None:
            single_thread_time = t
        speedup_1t = single_thread_time / t if single_thread_time else float("nan")
        speedup_np = t_np / t
        print(
            f"{threads:>8}  {t * 1000:>12.2f}  {gf:>10.2f}  "
            f"{speedup_1t:>13.2f}x  {speedup_np:>9.2f}x"
        )

    print("-" * 60)
    print()


if __name__ == "__main__":
    main()
