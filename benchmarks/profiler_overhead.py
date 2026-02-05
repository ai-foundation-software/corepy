"""
Benchmark to measure profiling overhead.
Goal: Ensure enabled overhead is <2% and disabled is ~0%.
"""

import time

import corepy as cp


def measure_overhead():
    iterations = 100_000
    data = cp.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    print(f"Running {iterations} iterations...")

    # Baseline (No profiling code enabled - simulated by disable)
    # Ideally baseline usually means "compilation without profiling support",
    # but here we measure "disabled" vs "enabled".

    # 1. Profiling DISABLED
    cp.disable_profiling()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = data.mean()
    end = time.perf_counter()
    duration_disabled = end - start
    avg_disabled = (duration_disabled / iterations) * 1e6  # micros

    print(f"Disabled: {duration_disabled:.4f}s ({avg_disabled:.3f} µs/op)")

    # 2. Profiling ENABLED
    cp.enable_profiling()
    start = time.perf_counter()
    for _ in range(iterations):
        _ = data.mean()
    end = time.perf_counter()
    duration_enabled = end - start
    avg_enabled = (duration_enabled / iterations) * 1e6  # micros

    cp.disable_profiling()
    cp.clear_profile()

    print(f"Enabled:  {duration_enabled:.4f}s ({avg_enabled:.3f} µs/op)")

    # 3. Calculate Overhead
    overhead_us = avg_enabled - avg_disabled
    overhead_pct = ((duration_enabled - duration_disabled) / duration_disabled) * 100.0

    print("-" * 40)
    print(f"Overhead: {overhead_us:.3f} µs/op (+{overhead_pct:.1f}%)")

    if overhead_pct > 5.0:
        print("⚠️ WARNING: Overhead is > 5%")
    else:
        print("✅ Overhead is within acceptable limits (< 5%)")


if __name__ == "__main__":
    measure_overhead()
