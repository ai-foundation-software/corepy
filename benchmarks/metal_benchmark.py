import time

import numpy as np

import corepy as cp
from corepy import array
from corepy.backend import BackendType, get_system_capabilities


def benchmark_op(name, func_np, func_cp, runs=5):
    # Warmup
    try:
        func_np()
        func_cp()
    except Exception as e:
        print(f"{name}: FAILED during warmup - {e}")
        return 0.0

    # NumPy
    start = time.perf_counter()
    for _ in range(runs):
        func_np()
    np_time = (time.perf_counter() - start) / runs * 1000

    # CorePy
    # Ensure synchrony if needed (Metal is async, but data transfer back forces wait usually)
    # Ideally should use device sync, but standard usage involves returning array which might not block.
    # To measure kernel time, we might need to force sync.
    # But for end-to-end user latency, we measure python-side return.
    # CorePy implementation of ops like 'add' returns a new array.
    # If the backend is fully async, this measures submission time.
    # But metal_backend.mm currently does `[cmdBuf waitUntilCompleted]` in dispatch functions!
    # So it IS synchronous.

    start = time.perf_counter()
    for _ in range(runs):
        func_cp()
    cp_time = (time.perf_counter() - start) / runs * 1000

    speedup = np_time / cp_time if cp_time > 0 else 0
    emoji = "🚀" if speedup > 1.0 else "⚠️"
    print(
        f"{name:<30} | NumPy: {np_time:8.3f} ms | CorePy: {cp_time:8.3f} ms | Speedup: {speedup:6.2f}x {emoji}"
    )
    return speedup


def main():
    caps = get_system_capabilities()
    if not caps.get("gpu", {}).get("metal_available", False):
        print("Metal GPU not available. Skipping benchmark.")
        return

    print("=== CorePy Metal Benchmark ===")

    # Sizes: Small (CPU wins), Medium (Crossover?), Large (GPU wins)
    sizes = [512, 1024, 2048, 4096]

    for N in sizes:
        print(f"\n--- Size {N}x{N} ({N * N * 4 / 1024 / 1024:.1f} MB) ---")
        shape = (N, N)
        try:
            a_np = np.random.rand(*shape).astype(np.float32)
            b_np = np.random.rand(*shape).astype(np.float32)

            a_cp = array(a_np, device="gpu")
            b_cp = array(b_np, device="gpu")

            # Add (Bandwidth bound)
            benchmark_op(
                f"Add {N}x{N}",
                lambda a=a_np, b=b_np: a + b,
                lambda a=a_cp, b=b_cp: a + b,
            )

            # Transpose (Bandwidth bound + stride)
            benchmark_op(f"Transpose {N}x{N}", lambda a=a_np: a.T, lambda a=a_cp: a.T)

            # Matmul (Compute bound)
            # Scaling is O(N^3), so limit runs/size
            if N <= 2048:
                benchmark_op(
                    f"Matmul {N}x{N}",
                    lambda a=a_np, b=b_np: a @ b,
                    lambda a=a_cp, b=b_cp: a @ b,
                    runs=3,
                )
            else:
                print(f"Matmul {N}x{N}: Skipped (too large)")

        except Exception as e:
            print(f"Skipping size {N}: {e}")


if __name__ == "__main__":
    main()
