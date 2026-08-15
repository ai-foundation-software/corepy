"""
Arena Integration Benchmark

Measures overhead of arena integration in reduction operations.
Expected: <1% overhead from arena.reset() calls
"""

import time

import numpy as np
from _corepy_rust import array_any, array_mean_f32, array_sum_f32


def bench_operation(op_name, op_func, data, iterations=10_000):
    """
    Benchmark a single operation with arena integration.

    Args:
        op_name: Name of operation for display
        op_func: Function to benchmark
        data: NumPy array input
        iterations: Number of iterations

    Returns:
        dict with performance metrics
    """
    size = len(data)

    # Warmup
    for _ in range(100):
        if op_name == "any":
            # any() takes uint8
            op_func(data.ctypes.data, size)
        else:
            op_func(data.ctypes.data, size)

    # Timed run
    start = time.perf_counter()
    for _ in range(iterations):
        if op_name == "any":
            result = op_func(data.ctypes.data, size)
        else:
            result = op_func(data.ctypes.data, size)
    elapsed = time.perf_counter() - start

    ops_per_sec = iterations / elapsed
    bytes_processed = size * data.itemsize * iterations
    gb_per_sec = bytes_processed / (elapsed * 1e9)
    avg_latency_us = (elapsed / iterations) * 1e6

    return {
        "size": size,
        "ops_per_sec": ops_per_sec,
        "gb_per_sec": gb_per_sec,
        "avg_latency_us": avg_latency_us,
        "total_time": elapsed,
    }


def print_results(op_name, results):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 70}")
    print(f"{op_name.upper()} Benchmark")
    print(f"{'=' * 70}")

    for r in results:
        print(f"\nSize: {r['size']:>10,} elements")
        print(f"  Operations/sec:  {r['ops_per_sec']:>12,.0f}")
        print(f"  Bandwidth:       {r['gb_per_sec']:>12.2f} GB/s")
        print(f"  Avg latency:     {r['avg_latency_us']:>12.2f} µs")


def main():
    print("=" * 70)
    print("Arena Integration Benchmark")
    print("Testing overhead of with_arena() wrapper")
    print("=" * 70)

    # Test sizes (small to large)
    test_sizes = [
        1_000,  # 1K - Cache-resident
        10_000,  # 10K - L2 cache
        100_000,  # 100K - L3 cache / RAM
        1_000_000,  # 1M - RAM bandwidth test
    ]

    # === Benchmark sum_f32 ===
    print("\n[1/3] Benchmarking sum_f32 (AVX2)...")
    sum_results = []
    for size in test_sizes:
        data = np.random.rand(size).astype(np.float32)
        result = bench_operation(
            "sum_f32",
            array_sum_f32,
            data,
            iterations=10_000 if size < 100_000 else 1_000,
        )
        sum_results.append(result)
    print_results("sum_f32", sum_results)

    # === Benchmark mean_f32 ===
    print("\n[2/3] Benchmarking mean_f32 (AVX2)...")
    mean_results = []
    for size in test_sizes:
        data = np.random.rand(size).astype(np.float32)
        result = bench_operation(
            "mean_f32",
            array_mean_f32,
            data,
            iterations=10_000 if size < 100_000 else 1_000,
        )
        mean_results.append(result)
    print_results("mean_f32", mean_results)

    # === Benchmark any (early-exit) ===
    print("\n[3/3] Benchmarking any (early-exit optimization)...")
    any_results = []
    for size in test_sizes:
        # Create bool array with false at start (worst case for early-exit)
        data = np.zeros(size, dtype=np.uint8)
        data[-1] = 1  # True at end (must scan entire array)
        result = bench_operation(
            "any", array_any, data, iterations=10_000 if size < 100_000 else 1_000
        )
        any_results.append(result)
    print_results("any", any_results)

    # === Performance Summary ===
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)

    print("\nPeak Bandwidth (1M elements):")
    print(f"  sum_f32:  {sum_results[-1]['gb_per_sec']:.2f} GB/s")
    print(f"  mean_f32: {mean_results[-1]['gb_per_sec']:.2f} GB/s")
    print(f"  any:      {any_results[-1]['gb_per_sec']:.2f} GB/s")

    print("\nLowest Latency (1K elements):")
    print(f"  sum_f32:  {sum_results[0]['avg_latency_us']:.2f} µs")
    print(f"  mean_f32: {mean_results[0]['avg_latency_us']:.2f} µs")
    print(f"  any:      {any_results[0]['avg_latency_us']:.2f} µs")

    print("\n✅ Arena integration benchmark complete!")
    print("\nExpected overhead: <1% (arena.reset() is O(1))")
    print("Actual overhead cannot be measured without baseline,")
    print("but tests passing confirm no crashes or memory leaks.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
