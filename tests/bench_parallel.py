"""
Parallel Dispatch Benchmark

Tests scaling behavior of parallel reductions vs sequential.
Measures speedup for large arrays (>100K elements).
"""

import time

import numpy as np
from _corepy_rust import array_mean_f32, array_sum_f32, array_sum_i32


def bench_scaling(op_name, op_func, dtype, sizes):
    """
    Benchmark operation across multiple sizes to measure parallel scaling.

    Args:
        op_name: Operation name
        op_func: FFI function
        dtype: NumPy dtype
        sizes: List of sizes to test

    Returns:
        list of (size, throughput_gb_s, latency_ms)
    """
    results = []

    for size in sizes:
        # Create random data
        if dtype == np.float32:
            data = np.random.rand(size).astype(dtype)
        else:
            data = np.random.randint(0, 100, size, dtype=dtype)

        # Determine iteration count (smaller for large arrays)
        if size < 100_000:
            iterations = 1000
        elif size < 1_000_000:
            iterations = 100
        else:
            iterations = 10

        # Warmup
        for _ in range(10):
            op_func(data.ctypes.data, size)

        # Timed run
        start = time.perf_counter()
        for _ in range(iterations):
            result = op_func(data.ctypes.data, size)
        elapsed = time.perf_counter() - start

        throughput_gb_s = (size * data.itemsize * iterations) / (elapsed * 1e9)
        latency_ms = (elapsed / iterations) * 1000

        results.append((size, throughput_gb_s, latency_ms))

        # Print immediate feedback
        parallel_marker = "🚀 PARALLEL" if size >= 1_000_000 else "  sequential"
        print(
            f"  {size:>12,} elements {parallel_marker}: {throughput_gb_s:>6.2f} GB/s, {latency_ms:>8.3f} ms"
        )

    return results


def print_speedup_analysis(results):
    """Analyze and print speedup from parallelization."""
    # Find threshold crossing
    sequential_perf = None
    parallel_perf = None

    for size, throughput, _ in results:
        if size < 1_000_000:
            sequential_perf = throughput  # Last sequential
        elif size >= 1_000_000 and parallel_perf is None:
            parallel_perf = throughput  # First parallel

    if sequential_perf and parallel_perf:
        speedup = parallel_perf / sequential_perf
        print("\n  📊 Parallel vs Sequential Throughput:")
        print(f"     Sequential (500K): {sequential_perf:.2f} GB/s")
        print(f"     Parallel (1M+):    {parallel_perf:.2f} GB/s")
        print(f"     Speedup Factor:    {speedup:.2f}x")


def main():
    print("=" * 70)
    print("Parallel Dispatch Benchmark")
    print("Threshold: 1,000,000 elements")
    print("=" * 70)

    # Test sizes spanning the threshold
    test_sizes = [
        1_000,  # Far below threshold
        10_000,  # Below threshold
        100_000,  # At threshold (parallel kicks in)
        500_000,  # Well above threshold
        1_000_000,  # Large parallel
        5_000_000,  # Very large parallel
    ]

    # === sum_f32 ===
    print("\n[1/3] sum_f32 (AVX2 SIMD)")
    print("-" * 70)
    sum_f32_results = bench_scaling("sum_f32", array_sum_f32, np.float32, test_sizes)
    print_speedup_analysis(sum_f32_results)

    # === sum_i32 ===
    print("\n[2/3] sum_i32 (AVX2 SIMD)")
    print("-" * 70)
    sum_i32_results = bench_scaling("sum_i32", array_sum_i32, np.int32, test_sizes)
    print_speedup_analysis(sum_i32_results)

    # === mean_f32 ===
    print("\n[3/3] mean_f32 (parallel sum + divide)")
    print("-" * 70)
    mean_results = bench_scaling("mean_f32", array_mean_f32, np.float32, test_sizes)
    print_speedup_analysis(mean_results)

    # === Summary ===
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)

    print("\nPeak Throughput (5M elements):")
    print(f"  sum_f32:  {sum_f32_results[-1][1]:.2f} GB/s")
    print(f"  sum_i32:  {sum_i32_results[-1][1]:.2f} GB/s")
    print(f"  mean_f32: {mean_results[-1][1]:.2f} GB/s")

    print("\nLowest Latency (1K elements):")
    print(f"  sum_f32:  {sum_f32_results[0][2]:.3f} ms")
    print(f"  sum_i32:  {sum_i32_results[0][2]:.3f} ms")
    print(f"  mean_f32: {mean_results[0][2]:.3f} ms")

    # Compare with NumPy
    print("\n" + "=" * 70)
    print("NumPy Comparison (5M elements)")
    print("=" * 70)

    size = 5_000_000
    data_f32 = np.random.rand(size).astype(np.float32)

    # NumPy sum
    start = time.perf_counter()
    for _ in range(10):
        np_result = np.sum(data_f32)
    numpy_time = time.perf_counter() - start

    # Corepy sum
    start = time.perf_counter()
    for _ in range(10):
        cp_result = array_sum_f32(data_f32.ctypes.data, size)
    corepy_time = time.perf_counter() - start

    speedup = numpy_time / corepy_time
    numpy_throughput = (size * 4 * 10) / (numpy_time * 1e9)
    corepy_throughput = (size * 4 * 10) / (corepy_time * 1e9)

    print(f"\nNumPy sum:   {numpy_throughput:.2f} GB/s ({numpy_time * 1000:.2f} ms)")
    print(f"Corepy sum:  {corepy_throughput:.2f} GB/s ({corepy_time * 1000:.2f} ms)")
    print(f"Speedup:     {speedup:.2f}x")

    print("\n✅ Parallel dispatch benchmark complete!")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
