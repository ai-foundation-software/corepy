"""
New Features Demonstration for Corepy 0.3.0
===========================================

This script demonstrates recently added features:
1. concatenate() - Joining arrays
2. compute_stats() - Efficient multi-statistic computation
"""

import time

import corepy as cp


def main():
    print("=" * 60)
    print("COREPY NEW FEATURES DEMO")
    print("=" * 60)
    print(f"Version: {cp.__version__}\n")

    # -------------------------------------------------------------------------
    # 1. Concatenate
    # -------------------------------------------------------------------------
    print("-" * 60)
    print("1. Concatenate (joining arrays)")
    print("-" * 60)

    a = cp.array([1.0, 2.0, 3.0])
    b = cp.array([4.0, 5.0, 6.0])

    print(f"ndarray a: {a}")
    print(f"ndarray b: {b}")

    # Concatenate 1D
    c = cp.concatenate((a, b))
    print(f"concatenated: {c}")
    print(f"shape: {c.shape}")

    # Concatenate 2D
    m1 = cp.array([[1, 2], [3, 4]])
    m2 = cp.array([[5, 6]])

    print(f"\nMatrix 1:\n{m1}")
    print(f"Matrix 2:\n{m2}")

    m3 = cp.concatenate((m1, m2), axis=0)
    print(f"concatenated (axis=0):\n{m3}")

    try:
        m4 = cp.concatenate((m1, m2.T), axis=1)
        print(f"concatenated (axis=1):\n{m4}")
    except Exception as e:
        print(f"Concatenate axis=1 failed (expected if shapes mismatch): {e}")

    # -------------------------------------------------------------------------
    # 2. Compute Stats
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("2. Compute Stats (multi-stat optimization)")
    print("-" * 60)

    data = cp.arange(0, 1000, 1, dtype=cp.Float32)
    print(f"Data size: {data.shape}")

    # Compute multiple stats in one call (simulated)
    stats = cp.compute_stats(data, ["mean", "std", "min", "max"])
    print(f"Stats computed: {stats}")

    print(f"Mean: {stats['mean']}")
    print(f"Min:  {stats['min']}")
    print(f"Max:  {stats['max']}")

    print("\n" + "=" * 60)
    print("✅ Demo Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
