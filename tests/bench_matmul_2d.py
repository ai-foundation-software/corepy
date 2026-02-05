import time

import numpy as np

import corepy as cp


def bench_matmul_2d():
    print("=" * 70)
    print("2D Matrix Multiplication Benchmark (Optimized AVX2)")
    print("=" * 70)

    test_sizes = [
        (100, 100),
        (256, 256),
        (512, 512),
        (1024, 1024),
    ]

    for m, n in test_sizes:
        k = n
        print(f"\nSize: {m} x {k} x {n}")

        a_np = np.random.rand(m, k).astype(np.float32)
        b_np = np.random.rand(k, n).astype(np.float32)

        a_cp = cp.Tensor(a_np)
        b_cp = cp.Tensor(b_np)

        iterations = 50 if m < 512 else 10

        # Warmup
        for _ in range(5):
            _ = a_cp.matmul(b_cp)
            _ = a_np @ b_np

        # NumPy
        start = time.perf_counter()
        for _ in range(iterations):
            expected = a_np @ b_np
        numpy_time = (time.perf_counter() - start) / iterations

        # Corepy
        start = time.perf_counter()
        for _ in range(iterations):
            result = a_cp.matmul(b_cp)
        corepy_time = (time.perf_counter() - start) / iterations

        # GFLOPS: 2 * m * n * k operations
        flops = 2 * m * n * k
        numpy_gflops = (flops / numpy_time) / 1e9
        corepy_gflops = (flops / corepy_time) / 1e9

        speedup = numpy_time / corepy_time

        # Accuracy check
        result_np = np.array(result._backing_data).reshape(m, n)
        error = np.max(np.abs(result_np - expected)) / (np.max(np.abs(expected)) + 1e-9)

        print(
            f"  Corepy (AVX2):   {corepy_time * 1000:>8.3f} ms  ({corepy_gflops:>6.2f} GFLOPS)"
        )
        print(
            f"  NumPy (OpenBLAS):{numpy_time * 1000:>8.3f} ms  ({numpy_gflops:>6.2f} GFLOPS)"
        )
        print(f"  Speedup:          {speedup:>6.2f}x")
        print(f"  Max Rel Error:    {error:>8.2e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    bench_matmul_2d()
