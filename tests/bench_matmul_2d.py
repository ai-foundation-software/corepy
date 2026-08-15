import argparse
import time

import numpy as np

import corepy as cp


def benchmark_matmul(m: int, k: int, n: int, num_iters: int = 10):
    print(f"Benchmarking 2D Matmul: ({m}x{k}) @ ({k}x{n})")

    # Setup
    a_np = np.random.randn(m, k).astype(np.float32)
    b_np = np.random.randn(k, n).astype(np.float32)

    a_cp = cp.ndarray(a_np)
    b_cp = cp.ndarray(b_np)

    # Warmup
    _ = a_cp.matmul(b_cp)
    _ = a_np @ b_np

    # NumPy Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        res_np = a_np @ b_np
    end = time.perf_counter()
    np_time = (end - start) / num_iters
    np_gflops = (2 * m * k * n) / (np_time * 1e9)

    # Corepy Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        res_cp = a_cp.matmul(b_cp)
    end = time.perf_counter()
    cp_time = (end - start) / num_iters
    cp_gflops = (2 * m * k * n) / (cp_time * 1e9)

    # Verification
    if hasattr(res_cp, "to_list()") and isinstance(res_cp.to_list(), np.ndarray):
        res_cp_np = res_cp.to_list()
    else:
        # Fallback for list-backed arrays
        res_cp_np = np.array(res_cp.to_list()).reshape(m, n)

    diff = np.abs(res_cp_np - res_np).max()
    is_correct = diff < 1e-4

    print(f"NumPy:  {np_time * 1000:8.4f} ms | {np_gflops:8.4f} GFLOPS")
    print(f"Corepy: {cp_time * 1000:8.4f} ms | {cp_gflops:8.4f} GFLOPS")
    print(f"Speedup: {np_time / cp_time:8.4f}x")
    print(f"Correctness: {'✅' if is_correct else '❌ (Diff: ' + str(diff) + ')'}")
    print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    benchmark_matmul(args.m, args.k, args.n, args.iters)
    # Test smaller sizes too
    benchmark_matmul(128, 128, 128, 20)
    benchmark_matmul(64, 64, 64, 50)
