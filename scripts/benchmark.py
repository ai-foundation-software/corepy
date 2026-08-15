import sys
import time

import numpy as np

import corepy


def run_benchmark():
    print("=== Performance Test ===")
    try:
        policy = corepy.get_backend_policy()
        print(f"Backend: {policy}")
    except Exception as e:
        print(f"Backend: Unknown ({e})")
    print("")

    # Test different sizes
    sizes = [(100, 100), (512, 512), (1024, 1024)]

    for m, n in sizes:
        try:
            # Create random arrays
            a_np = np.random.randn(m, n).astype(np.float32)
            b_np = np.random.randn(n, m).astype(np.float32)

            a = corepy.array(a_np)
            b = corepy.array(b_np)

            # Warmup
            _ = a.matmul(b)

            # Benchmark
            start = time.time()
            iterations = 10
            for _ in range(iterations):
                c = a.matmul(b)
            elapsed = (time.time() - start) / iterations

            # Get dispatch info if available
            try:
                dispatch_info = corepy.explain_last_dispatch()
            except Exception:
                dispatch_info = "N/A"

            print(f"{m}x{n}: {elapsed * 1000:.2f}ms - {dispatch_info}")

        except Exception as e:
            print(f"{m}x{n}: FAILED - {e}")

    print("")
    print("=== Performance test complete ===")


if __name__ == "__main__":
    run_benchmark()
