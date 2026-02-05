import time

import numpy as np

import corepy


def benchmark_compare():
    """compare CorePy vs NumPy performance"""
    sizes = [(64, 64), (256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]

    print(
        f"{'Size':<15} | {'NumPy (ms)':<12} | {'CorePy (ms)':<12} | {'Speedup':<10} | {'Backend/Policy'}"
    )
    print("-" * 80)

    policies = [
        ("Default", corepy.BackendPolicy.DEFAULT),
        ("BLAS", corepy.BackendPolicy.BLAS),
    ]

    for policy_name, policy in policies:
        print(f"--- Policy: {policy_name} ---")
        corepy.set_backend_policy(policy)

        for m, n in sizes:
            # Data generation
            np_a = np.random.randn(m, n).astype(np.float32)
            np_b = np.random.randn(n, m).astype(np.float32)

            cp_a = corepy.Tensor(np_a)
            cp_b = corepy.Tensor(np_b)

            # --- NumPy Benchmark ---
            # Warmup
            _ = np_a @ np_b

            times_np = []
            for _ in range(5):
                start = time.perf_counter()
                _ = np_a @ np_b
                times_np.append(time.perf_counter() - start)
            avg_np = np.mean(times_np) * 1000  # ms

            # --- CorePy Benchmark ---
            # Warmup
            _ = cp_a.matmul(cp_b)

            times_cp = []
            for _ in range(5):
                start = time.perf_counter()
                _ = cp_a.matmul(cp_b)
                times_cp.append(time.perf_counter() - start)
            avg_cp = np.mean(times_cp) * 1000  # ms

            speedup = avg_np / avg_cp

            # Get backend used
            dispatch_info = corepy.explain_last_dispatch()
            # Extract backend name locally for cleaner table if possible, but full string is good too.
            # Let's keep it somewhat concise.
            backend_short = dispatch_info.split("(")[
                0
            ].strip()  # e.g. "matmul → OpenBLAS"

            print(
                f"{m}x{n:<10} | {avg_np:<12.2f} | {avg_cp:<12.2f} | {speedup:<10.2f} | {backend_short}"
            )


if __name__ == "__main__":
    benchmark_compare()
