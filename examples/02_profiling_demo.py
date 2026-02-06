import time

import corepy as cp
from corepy.backend import BackendPolicy, set_backend_policy


def profile_workload():
    """
    Shows how to use the built-in profiler to measure performance
    and detect bottlenecks.
    """
    print("--- Corepy Profiling Demo ---")

    # 1. Start the profiling context
    # This hooks into the Rust dispatcher to record high-precision timings
    cp.enable_profiling()

    # 2. Setup Data (Large enough to matter)
    # 1 Million elements
    size = 1_000_000
    print(f"Allocating {size} elements...")
    data = cp.Tensor([1.0] * size)

    # 3. Workload
    print("Running optimized kernels...")

    # Simple Math
    v1 = data + 10.0
    v2 = v1 * 0.5

    # Reduction
    result = v2.sum()
    print(f"Calculation Result: {result}")

    # 4. Stop and Report
    cp.disable_profiling()
    print("\n--- Profile Report ---")
    report = cp.profile_report()
    print(report)


if __name__ == "__main__":
    profile_workload()
