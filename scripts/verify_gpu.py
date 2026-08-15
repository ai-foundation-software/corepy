import time

import numpy as np

import corepy as cp


def test_auto_gpu():
    print("=== GPU Auto-Detection Verification ===")

    # 1. Check system capabilities
    caps = cp.get_system_capabilities()
    print(f"Detected Capabilities: {caps}")

    # 2. Test large matmul for auto-dispatch
    # A size that should trigger the GPU threshold (v0.3.0 default is flops > 50M)
    # 512x512x512 matmul is approx 2 * 512^3 = 268M flops
    N = 512
    a = cp.array(np.random.randn(N, N).astype(np.float32))
    b = cp.array(np.random.randn(N, N).astype(np.float32))

    start = time.perf_counter()
    c = a.matmul(b)
    end = time.perf_counter()

    print(f"Matmul {N}x{N} took {(end - start) * 1000:.2f}ms")
    print(f"Last Dispatch: {cp.explain_last_dispatch()}")

    # 3. Test explicit policy
    if caps["gpu"]["cuda_available"]:
        print("\nTesting Explicit CUDA Policy...")
        cp.set_backend_policy(cp.BackendPolicy.CUDA)
        c = a.matmul(b)
        print(f"Explicit CUDA Dispatch: {cp.explain_last_dispatch()}")

    if caps["gpu"]["metal_available"]:
        print("\nTesting Explicit Metal Policy...")
        cp.set_backend_policy(cp.BackendPolicy.METAL)
        c = a.matmul(b)
        print(f"Explicit Metal Dispatch: {cp.explain_last_dispatch()}")

    # 4. Verify Correctness (small scale)
    n = 4
    a_small = cp.ndarray(np.ones((n, n), dtype=np.float32))
    b_small = cp.ndarray(np.ones((n, n), dtype=np.float32))
    res = a_small.matmul(b_small)
    expected = n  # all ones matmul result is row/col sum
    if abs(res[0, 0] - expected) < 1e-5:
        print("\n✅ Correctness check passed.")
    else:
        print(f"\n❌ Correctness check failed. Got {res[0, 0]}, expected {expected}")


if __name__ == "__main__":
    test_auto_gpu()
