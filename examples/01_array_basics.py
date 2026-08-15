import time

import corepy as cp


def demo_array_basics():
    """
    Demonstrates basic array creation and arithmetic.
    """
    print("--- 1. ndarray Creation ---")
    # Arrays are strictly typed (default Float32)
    t1 = cp.ndarray([1.0, 2.0, 3.0, 4.0])
    print(f"t1: {t1} (Backend: {t1.backend.name})")

    # Creating from a list of lists works too
    t2 = cp.ndarray([[1, 2], [3, 4]])
    print(f"t2 Shape: {t2.shape}")

    print("\n--- 2. Arithmetic ---")
    # Element-wise operations
    doubled = t1 * 2.0
    print(f"t1 * 2.0 = {doubled}")

    squared = t1 * t1
    print(f"t1 * t1  = {squared}")

    print("\n--- 3. Reductions ---")
    # Highly optimized SIMD reductions
    print(f"Sum:  {t1.sum()}")
    print(f"Mean: {t1.mean()}")
    print(f"Max:  {t1.max()}")


if __name__ == "__main__":
    demo_array_basics()
