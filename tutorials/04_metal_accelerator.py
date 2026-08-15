"""
Tutorial 04: Metal GPU Acceleration
================================

This tutorial demonstrates how to use the Metal GPU backend on macOS devices (Apple Silicon).
Corepy v0.3.0 introduces native Metal support for high-performance array operations.


Prerequisites:
- macOS 12.0+
- Apple Silicon (M1/M2/M3) recommended

concepts:
- Device selection
- GPU vs CPU performance comparison
- Automatic fallback
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
        # Force synchronization if needed (currently synchronous)
    end = time.perf_counter()

    avg_time = (end - start) / iterations
    print(f"  {name}: {avg_time * 1000:.4f} ms")


def main():
    print("=== Corepy Metal GPU Tutorial ===")

    # 1. Check Availability
    caps = cp.get_system_capabilities()
    has_metal = caps.get("gpu", {}).get("metal_available", False)

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

    # 3. Metal GPU Execution
    print("\n--- Metal Backend ---")
    if not has_metal:
        print("Note: Metal is not available on this system. Skipping Metal benchmarks.")
        return

    try:
        # Create arrays directly on Metal device
        t_metal_a = cp.array(data, device="metal")
        t_metal_b = cp.array(data, device="metal")

        # Verify device
        # Note: If Metal is unavailable, it might have fallen back to CPU
        print(f"ndarray device: {t_metal_a.backend.value}")

        benchmark_op(
            "Matmul (Metal)", t_metal_a, t_metal_b, lambda x, y: x @ y, iterations=20
        )

        # 4. Mixed Operations
        print("\n--- Mixed Operations ---")
        # Adding Metal array to CPU array -> Result is usually on Metal (or CPU depending on promotion rules)
        # Currently Corepy might enforce same-device policies, so let's try strict

        res = t_metal_a + t_metal_b
        print(f"Result shape: {res.shape}")

    except Exception as e:
        print(f"\n⚠️ Metal backend unavailable or failed: {e}")
        print("Note: This tutorial requires a macOS device with Metal support.")


if __name__ == "__main__":
    main()
