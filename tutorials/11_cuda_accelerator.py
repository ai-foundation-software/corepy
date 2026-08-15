"""
Tutorial 11: CUDA GPU Acceleration
================================

This tutorial demonstrates how to use the CUDA GPU backend on Windows/Linux devices.
Corepy v0.3.0 introduces native CUDA support for high-performance array operations.

concepts:
- Device selection for NVIDIA hardware
- GPU vs CPU performance comparison
- Automatic fallback mechanisms
"""

import time

import numpy as np

import corepy as cp


def benchmark_op(name, array_a, array_b, op_func, iterations=10):
    # Warmup
    op_func(array_a, array_b)

    start = time.perf_counter()
    for _ in range(iterations):
        res = op_func(array_a, array_b)
        # Force synchronization if needed
    end = time.perf_counter()

    avg_time = (end - start) / iterations
    print(f"  {name}: {avg_time * 1000:.4f} ms")


def main():
    print("=== Corepy CUDA GPU Tutorial ===")

    # 1. Check Availability
    caps = cp.get_system_capabilities()
    has_cuda = caps.get("gpu", {}).get("cuda_available", False)

    print("\n1. Initializing Arrays...")
    N = 2048
    print(f"Problem size: {N}x{N} matrix ({N * N * 4 / 1024**2:.2f} MB)")

    # Create data in NumPy (CPU)
    data = np.random.randn(N, N).astype(np.float32)

    # 2. CPU Execution
    print("\n--- CPU Backend ---")
    t_cpu_a = cp.array(data)  # Default is CPU
    t_cpu_b = cp.array(data)

    benchmark_op("Matmul (CPU)", t_cpu_a, t_cpu_b, lambda x, y: x @ y, iterations=5)

    # 3. CUDA GPU Execution
    print("\n--- CUDA Backend ---")
    if not has_cuda:
        print("Note: CUDA is not available on this system. Skipping CUDA benchmarks.")
        return

    try:
        # Create arrays directly on CUDA device
        t_cuda_a = cp.array(data, device="cuda")
        t_cuda_b = cp.array(data, device="cuda")

        print(f"ndarray device: {t_cuda_a.backend.value}")

        benchmark_op(
            "Matmul (CUDA)", t_cuda_a, t_cuda_b, lambda x, y: x @ y, iterations=20
        )

        # 4. Mixed Operations
        print("\n--- Mixed Operations ---")
        res = t_cuda_a + t_cuda_b
        print(f"Result shape: {res.shape}")

    except Exception as e:
        print(f"\n⚠️ CUDA backend explicitly failed: {e}")


if __name__ == "__main__":
    main()
