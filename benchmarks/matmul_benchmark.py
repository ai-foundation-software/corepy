# ~/VSCode/corepy/benchmark.py
import time

import numpy as np

import corepy


def benchmark_matmul():
    """Benchmark different matrix sizes and backends"""
    sizes = [(64, 64), (256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]

    print("Matrix Multiplication Benchmark")
    print("=" * 60)

    for policy_name, policy in [
        ("Default", corepy.BackendPolicy.DEFAULT),
        ("BLAS", corepy.BackendPolicy.BLAS),
    ]:
        print(f"\nPolicy: {policy_name}")
        print("-" * 40)

        corepy.set_backend_policy(policy)

        for m, n in sizes:
            # Create random matrices
            a = corepy.Tensor(np.random.randn(m, n).astype(np.float32))
            b = corepy.Tensor(np.random.randn(n, m).astype(np.float32))

            # Warmup
            _ = a.matmul(b)

            # Benchmark
            times = []
            for _ in range(5):
                start = time.perf_counter()
                c = a.matmul(b)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            gflops = (2 * m * n * m) / (avg_time * 1e9)

            print(
                f"  {m}x{n}: {avg_time * 1000:.2f}ms ({gflops:.2f} GFLOPs) - {corepy.explain_last_dispatch()}"
            )


if __name__ == "__main__":
    benchmark_matmul()
