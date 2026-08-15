import time

import numpy as np

# Import internal FFI function for direct testing
from _corepy_rust import array_dot_product_f32

import corepy as cp
from corepy.backend import detect_devices


def bench_dot_product(size=1_000_000, iterations=1000):
    """
    Benchmark Corepy Dot Product vs NumPy
    """
    print(f"\nBenchmarking size: {size:,} elements")

    # Data Setup
    data_np = np.random.rand(size).astype(np.float32)
    data_cp_a = cp.ndarray(data_np.tolist())
    data_cp_b = cp.ndarray(data_np.tolist())

    # Pointers for direct FFI call
    ptr_a = data_np.ctypes.data
    ptr_b = (
        data_np.ctypes.data
    )  # Use same data for simplicity/cache locality or separate if strict

    # 1. Warmup & Correctness Check
    res_np = np.dot(data_np, data_np)
    res_cp = array_dot_product_f32(ptr_a, ptr_b, size)

    # Allow small FP error for different accumulation order/SIMD
    if not np.isclose(res_np, res_cp, rtol=1e-4):
        print(f"❌ Mismatch! NumPy: {res_np}, Corepy: {res_cp}")
        return
    else:
        print(f"✅ Correctness verified (Val: {res_cp:.4f})")

    # 2. NumPy Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        np.dot(data_np, data_np)
    dt_np = time.perf_counter() - start

    # 3. Corepy Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        array_dot_product_f32(ptr_a, ptr_b, size)
    dt_cp = time.perf_counter() - start

    ops = size * 2 * iterations  # MAC = 2 ops
    print(f"NumPy Speed:  {ops / dt_np / 1e9:.2f} GFLOPS")
    print(f"Corepy Speed: {ops / dt_cp / 1e9:.2f} GFLOPS")
    print(f"Speedup:      {dt_np / dt_cp:.2f}x")


if __name__ == "__main__":
    print("=== SIMD Dot Product Benchmark ===")
    info = detect_devices()
    print(f"AVX2 Detected: {info.has_avx2}")

    bench_dot_product(size=10_000, iterations=10_000)
    bench_dot_product(size=100_000, iterations=1000)
    bench_dot_product(size=1_000_000, iterations=100)
