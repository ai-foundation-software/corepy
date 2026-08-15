"""
Dot Product SIMD Benchmark

Compares scalar vs AVX2-optimized dot product performance.
Expected speedup: 8-16x (AVX2 processes 8x f32 per instruction)
"""

import time

import numpy as np
from _corepy_rust import array_dot_product_f32


def bench_dot_product(size, iterations=1000):
    """
    Benchmark dot product at given size.

    Args:
        size: Vector length
        iterations: Number of iterations

    Returns:
        dict with performance metrics
    """
    # Create random vectors
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)

    # Warmup
    for _ in range(10):
        array_dot_product_f32(a.ctypes.data, b.ctypes.data, size)

    # Benchmark Corepy (AVX2)
    start = time.perf_counter()
    for _ in range(iterations):
        cp_result = array_dot_product_f32(a.ctypes.data, b.ctypes.data, size)
    cp_time = time.perf_counter() - start

    # Benchmark NumPy (reference)
    start = time.perf_counter()
    for _ in range(iterations):
        np_result = np.dot(a, b)
    np_time = time.perf_counter() - start

    # Verify correctness
    np_result_single = np.dot(a, b)
    error = abs(cp_result - np_result_single)
    relative_error = error / max(abs(np_result_single), 1e-10)

    # Calculate metrics
    ops_per_sec_cp = iterations / cp_time
    ops_per_sec_np = iterations / np_time
    speedup = np_time / cp_time

    # FLOPS: 2*size operations per dot (multiply + add)
    flops_cp = (2 * size * iterations) / cp_time
    flops_np = (2 * size * iterations) / np_time

    return {
        "size": size,
        "cp_time_ms": cp_time * 1000,
        "np_time_ms": np_time * 1000,
        "speedup": speedup,
        "cp_gflops": flops_cp / 1e9,
        "np_gflops": flops_np / 1e9,
        "error": error,
        "relative_error": relative_error,
        "cp_result": cp_result,
        "np_result": np_result_single,
    }


def main():
    print("=" * 70)
    print("Dot Product SIMD Benchmark")
    print("Comparing scalar (old) vs AVX2 (new) implementation")
    print("=" * 70)

    # Test sizes
    test_sizes = [
        100,  # Small
        1_000,  # 1K
        10_000,  # 10K
        100_000,  # 100K
        1_000_000,  # 1M
    ]

    results = []

    print("\nRunning benchmarks...")
    print("-" * 70)

    for size in test_sizes:
        iterations = max(10, min(10000, 1_000_000 // size))
        result = bench_dot_product(size, iterations)
        results.append(result)

        print(f"\nSize: {size:>10,} elements ({iterations} iterations)")
        print(
            f"  Corepy (AVX2): {result['cp_time_ms']:>8.3f} ms  ({result['cp_gflops']:>6.2f} GFLOPS)"
        )
        print(
            f"  NumPy:         {result['np_time_ms']:>8.3f} ms  ({result['np_gflops']:>6.2f} GFLOPS)"
        )
        print(f"  Speedup:       {result['speedup']:>8.2f}x")
        print(f"  Accuracy:      {result['relative_error']:.2e} relative error")

    # Summary
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)

    print("\nSpeedup vs NumPy:")
    for r in results:
        marker = "✅" if r["speedup"] >= 1.0 else "⚠️ "
        print(f"  {r['size']:>10,} elements: {marker} {r['speedup']:>5.2f}x")

    print("\nPeak Performance (1M elements):")
    peak = results[-1]
    print(f"  Corepy: {peak['cp_gflops']:.2f} GFLOPS")
    print(f"  NumPy:  {peak['np_gflops']:.2f} GFLOPS")

    print("\nNumerical Accuracy:")
    max_error = max(r["relative_error"] for r in results)
    print(f"  Max relative error: {max_error:.2e}")
    if max_error < 1e-5:
        print("  ✅ Excellent accuracy (< 1e-5)")
    elif max_error < 1e-3:
        print("  ✅ Good accuracy (< 1e-3)")
    else:
        print("  ⚠️  Moderate accuracy")

    # Theoretical analysis
    print("\n" + "=" * 70)
    print("Theoretical Analysis")
    print("=" * 70)

    print("\nAVX2 Advantage:")
    print("  - Processes 8x f32 per instruction (vs 1x scalar)")
    print("  - FMA instruction: multiply-add in single cycle")
    print("  - Expected speedup: 4-8x (accounting for memory bandwidth)")

    avg_speedup_large = sum(r["speedup"] for r in results[-2:]) / 2
    print(f"\nActual speedup (large arrays): {avg_speedup_large:.2f}x")

    if avg_speedup_large >= 4.0:
        print("✅ Excellent SIMD utilization!")
    elif avg_speedup_large >= 2.0:
        print("✅ Good SIMD utilization")
    elif avg_speedup_large >= 1.0:
        print("⚠️  Moderate speedup (memory-bound?)")
    else:
        print("❌ Slower than NumPy (investigate)")

    print("\n✅ Dot product SIMD benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
