"""
Tutorial 13: CORE-50 and Lazy Fusion

This script demonstrates how to leverage the newly implemented CORE-50
operations alongside CorePy's element-wise kernel fusion.
"""

import time

import corepy as cp
from corepy.lazy.array import LazyArray


def main():
    print("--- CORE-50 & Lazy Fusion Demo ---")
    size = 2_000_000

    print(f"\nAllocating array of size {size:,}...")
    # Use python list comprehension for initial data, or better yet, cp.ones!
    arr = cp.ones(size)
    # Multiply by an angle for trig functions
    arr = cp.multiply(arr, 3.14159 / 4)  # ~45 degrees (pi/4)

    # ==========================
    # 1. Eager Evaluation
    # ==========================
    start = time.perf_counter()
    # Calculates sin^2(x) + cos^2(x) eagerly (allocates 2 intermediate arrays)
    eager_result = cp.sin(arr) ** 2 + cp.cos(arr) ** 2
    eager_time = (time.perf_counter() - start) * 1000

    # Calculate error
    eager_mean = cp.mean(eager_result)

    print("\nEager Execution:")
    print(f"  Result Mean: {eager_mean:.6f} (Expected 1.0)")
    print(f"  Time taken:  {eager_time:.3f} ms")

    # ==========================
    # 2. Lazy Fusion
    # ==========================
    la = LazyArray(arr)

    start = time.perf_counter()
    # Builds the graph
    lazy_expr = (la.sin() ** 2) + (la.cos() ** 2)
    # Compiles graph, topologically sorts element-wise ops, and runs single pass!
    lazy_result = lazy_expr.compute()
    lazy_time = (time.perf_counter() - start) * 1000

    # Calculate error
    lazy_mean = lazy_result.mean()

    print("\nLazy Fused Execution:")
    print(f"  Result Mean: {lazy_mean:.6f} (Expected 1.0)")
    print(f"  Time taken:  {lazy_time:.3f} ms")

    # Improvement ratio
    ratio = eager_time / lazy_time if lazy_time > 0 else 0
    print(f"\nSpeedup from fusion: {ratio:.2f}x")


if __name__ == "__main__":
    main()
