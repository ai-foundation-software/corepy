import os
import sys
import time

import numpy as np

# Ensure we import the local corepy
sys.path.insert(0, os.getcwd())
import corepy
from corepy import array
from corepy.backend.types import BackendType

# Enable profiling
try:
    corepy.enable_profiling()
except Exception:
    pass


def benchmark(name, func, iter_count=10):
    # Warmup
    try:
        func()
    except Exception as e:
        print(f"FAILED during warmup: {e}")
        return

    # Sync if needed (mostly synchronous for now except internal Metal command buffer wait)

    start = time.perf_counter()
    for _ in range(iter_count):
        func()
    end = time.perf_counter()
    avg_ms = ((end - start) / iter_count) * 1000
    print(f"{name}: {avg_ms:.3f} ms")


def verify_metal_ops():
    print("=== Verifying Metal Operations ===")

    # Check if Metal available
    try:
        caps = corepy.get_system_capabilities()
        if not caps["gpu"]["metal_available"]:
            print("SKIP: Metal not available on this system")
            return

        print("Metal is available. Proceeding with verification.")

        # Test Broadcasting
        print("\nTesting Broadcasting (2x3 + 1x3)...")
        # Using corepy.array as cp.array is not defined in this context
        a = corepy.array([[1, 2, 3], [4, 5, 6]], dtype="float32", device="metal")
        b = corepy.array([[10, 20, 30]], dtype="float32", device="metal")
        try:
            c = a + b
            expected = np.array([[11, 22, 33], [14, 25, 36]], dtype="float32")
            if np.allclose(c.to_numpy(), expected, atol=1e-4):
                print("Broadcasting Passed!")
            else:
                print(
                    f"Broadcasting Failed!\nExpected:\n{expected}\nGot:\n{c.to_numpy()}"
                )
                sys.exit(1)
        except Exception as e:
            print(f"Broadcasting Failed with error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    except Exception as e:
        print(f"Warning: could not check metal availability: {e}")

    N = 2048  # Large enough to trigger Metal path (>1024)
    shape = (N, N)

    print(f"Creating arrays of shape {shape} on GPU...")
    try:
        a = array(np.random.rand(*shape).astype(np.float32), device="gpu")
        b = array(np.random.rand(*shape).astype(np.float32), device="gpu")
    except Exception as e:
        print(f"FAILED to create GPU array: {e}")
        return

    # 1. Binary Ops
    print("\n--- Binary Ops ---")
    try:
        c = a + b
        benchmark("Add (Metal)", lambda: a + b)

        # Verify correctness
        np_a = a.to_numpy()
        np_b = b.to_numpy()
        np_c = c.to_numpy()

        # CorePy initializes output with garbage or zero?
        # Metal kernels overwrite.

        diff = np.abs(np_a + np_b - np_c).max()
        if diff > 1e-3:
            print(f"FAILED: Add correctness. Max diff: {diff}")
        else:
            print(f"PASSED: Add correctness. Max diff: {diff}")

    except Exception as e:
        print(f"FAILED Add: {e}")

    # 2. Reductions
    print("\n--- Reductions ---")
    try:
        res_max = a.max()
        benchmark("Max (Metal)", lambda: a.max())

        np_max = np.max(np_a)
        core_max = res_max.to_numpy()[0]

        if abs(np_max - core_max) > 1e-3:
            print(f"FAILED: Max correctness. Expected {np_max}, got {core_max}")
        else:
            print("PASSED: Max correctness")
    except Exception as e:
        print(f"FAILED Max: {e}")

    # 3. Transpose
    print("\n--- Transpose ---")
    try:
        t = a.T
        benchmark("Transpose (Metal)", lambda: a.transpose())

        np_t = np_a.T
        core_t = t.to_numpy()

        diff = np.abs(np_t - core_t).max()
        if diff > 1e-3:
            print(f"FAILED: Transpose correctness. Max diff: {diff}")
        else:
            print(f"PASSED: Transpose correctness. Max diff: {diff}")
    except Exception as e:
        print(f"FAILED Transpose: {e}")

    # 4. Profile Report
    print("\n--- Profile Report ---")
    try:
        report = corepy.get_profile_report()
        print(report)
        # Check if "Metal" appears in report
        if "Metal" in report:
            print("SUCCESS: Metal kernels were dispatched.")
        else:
            print(
                "WARNING: 'Metal' tag not found in profile report. May have fallen back to CPU."
            )
    except Exception as e:
        print(f"Could not print profile report: {e}")


if __name__ == "__main__":
    verify_metal_ops()
