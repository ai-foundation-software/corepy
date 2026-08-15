#!/usr/bin/env python3
"""
Tutorial 12 — CPU Backend & Adaptive BLAS Selection
=====================================================

CorePy now automatically selects the best math backend for matrix
multiplication based on your CPU and available libraries:

  Intel CPU + MKL installed    → Intel MKL   (fastest for Intel)
  AMD CPU + AOCL installed     → AMD AOCL    (fastest for Zen)
  Apple Silicon (M1/M2/M3/M4) → Accelerate  (AMX + vecLib)
  Any CPU + OpenBLAS           → OpenBLAS    (solid fallback)
  No BLAS at all               → Pure Rust   (always available)

Thread counts are automatically adaptive:
  matrix < 256   →  1 thread   (avoids overhead entirely)
  256 – 1023     →  min(cores/2, 4)
  1024 – 2047    →  cores/2
  2048 – 4095    →  all physical cores
  ≥ 4096         →  all logical cores (incl. Hyperthreads)

Topics in this tutorial:
  1. Inspecting the selected backend
  2. Running matmul through the new API
  3. Manually selecting a backend
  4. Benchmarking vs NumPy
  5. Environment variable control
"""

import time

import numpy as np

# ─── Import the new corepy.matmul module ─────────────────────────────────────
import corepy.matmul as cm

# ==========================================================================
# 1. Inspect the selected backend
# ==========================================================================
print("=" * 60)
print("1. CPU Backend Auto-Detection")
print("=" * 60)

info = cm.backend_info()
print(f"  CPU brand     : {info['brand']}")
print(f"  Vendor        : {info['vendor']}")
print(f"  Physical cores: {info['physical_cores']}")
print(f"  Logical cores : {info['logical_cores']}")
print(f"  Hyperthreading: {info['hyperthreading']}")
print(f"  Backend chosen: {info['backend']}")
print(f"  Thread count  : {info['threads']}  (for 2048×2048 matmul)")
print()

# You can also use the COREPY_BACKEND environment variable to override:
#   COREPY_BACKEND=rust  python tutorials/12_cpu_backend.py

# ==========================================================================
# 2. Basic matmul through the new API
# ==========================================================================
print("=" * 60)
print("2. Basic Matrix Multiplication")
print("=" * 60)

rng = np.random.default_rng(42)

# Small matrix (< 256) — single-threaded SIMD path
A = rng.standard_normal((64, 64)).astype(np.float32)
B = rng.standard_normal((64, 64)).astype(np.float32)

C_corepy = cm.matmul(A, B)  # auto backend
C_numpy = np.matmul(A, B)

max_diff = np.max(np.abs(C_corepy - C_numpy))
print(f"  64×64 matmul — max abs diff vs NumPy: {max_diff:.2e}  ✓")
print(f"  dispatch: {cm.get_last_dispatch()}")
print()

# Large matrix — uses full thread pool
A_big = rng.standard_normal((1024, 1024)).astype(np.float32)
B_big = rng.standard_normal((1024, 1024)).astype(np.float32)

C_big = cm.matmul(A_big, B_big)
C_np_big = np.matmul(A_big, B_big)

max_diff_big = np.max(np.abs(C_big - C_np_big))
print(f"  1024×1024 matmul — max abs diff vs NumPy: {max_diff_big:.2e}  ✓")
print(f"  dispatch: {cm.get_last_dispatch()}")
print()

# ==========================================================================
# 3. Manually selecting a backend
# ==========================================================================
print("=" * 60)
print("3. Manual Backend Selection")
print("=" * 60)

A_m = rng.standard_normal((512, 512)).astype(np.float32)
B_m = rng.standard_normal((512, 512)).astype(np.float32)

for policy in ["auto", "openblas", "rust"]:
    C_m = cm.matmul(A_m, B_m, policy=policy)
    print(f"  policy={policy!r:<10}  dispatch → {cm.get_last_dispatch()}")

# Using BackendPolicy enum
from corepy.matmul import BackendPolicy

C_enum = cm.matmul(A_m, B_m, policy=BackendPolicy.RUST)
print(f"  policy=BackendPolicy.RUST  → {cm.get_last_dispatch()}")
print()

# ==========================================================================
# 4. Quick performance comparison vs NumPy
# ==========================================================================
print("=" * 60)
print("4. Performance vs NumPy  (median of 10 runs)")
print("=" * 60)

sizes = [128, 512, 1024, 2048]


def bench(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]  # median


print(
    f"  {'Size':>6}  {'NumPy ms':>10}  {'CorePy ms':>10}  {'Speedup':>8}  {'GFLOP/s':>8}"
)
print(f"  {'-' * 6}  {'-' * 10}  {'-' * 10}  {'-' * 8}  {'-' * 8}")

out = None
for n in sizes:
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)
    out = np.empty((n, n), dtype=np.float32)

    t_np = bench(lambda a=a, b=b: np.matmul(a, b))
    t_cp = bench(lambda a=a, b=b, out=out: cm.matmul(a, b, out=out))

    gflops = 2 * n**3 / t_cp / 1e9
    speedup = t_np / t_cp
    print(
        f"  {n:>6}  {t_np * 1000:>10.2f}  {t_cp * 1000:>10.2f}"
        f"  {speedup:>7.2f}x  {gflops:>7.2f}"
    )

print()

# ==========================================================================
# 5. Environment variable control
# ==========================================================================
print("=" * 60)
print("5. Environment Variable Reference")
print("=" * 60)

print("""
  # Override backend selection (set BEFORE importing corepy):
  COREPY_BACKEND=rust      COREPY_BACKEND=openblas
  COREPY_BACKEND=mkl       COREPY_BACKEND=aocl

  # Cap thread count (CorePy sets these automatically):
  MKL_NUM_THREADS=4        OPENBLAS_NUM_THREADS=4
  BLIS_NUM_THREADS=4       RAYON_NUM_THREADS=4
  OMP_NUM_THREADS=4        # Safety net for nested libs

  # MKL-specific:
  MKL_DYNAMIC=FALSE        # Disable MKL's own thread manager
  MKLROOT=/opt/intel/oneapi/mkl/latest

  # AOCL-specific:
  AOCL_ROOT=/opt/aocl
""")

print("Tutorial complete!")
