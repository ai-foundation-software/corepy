"""
Tutorial 09: Lazy Evaluation & Operation Fusion
===============================================
This tutorial demonstrates the powerful LazyArray feature in CorePy v0.3.0.
By delaying execution, CorePy can fuse multiple operations together
into highly optimized kernels, minimizing memory allocations
and boosting performance for complex math.
"""

import time

import corepy as cp
from corepy.lazy.array import LazyArray


def main():
    print("--- 1. Immediate Execution vs. Lazy Evaluation ---")

    # 1. Immediate (Eager) Evaluation
    # Each operation creates a new array in memory
    print("Eager (Immediate) Execution:")
    a = cp.array([1.0, 2.0, 3.0, 4.0])
    b = cp.array([5.0, 6.0, 7.0, 8.0])
    c = cp.array([9.0, 10.0, 11.0, 12.0])

    # Executed sequentially, creating intermediate arrays
    result_eager = (a + b) * c - (a / 2.0)
    print(f"Eager Result: {result_eager}\\n")

    # 2. Lazy Evaluation
    # We wrap arrays in LazyArray to build an expression tree
    print("Lazy Execution (Fusion):")
    lazy_a = LazyArray(a)
    lazy_b = LazyArray(b)
    lazy_c = LazyArray(c)

    # No actual computation happens here, we just build the tree!
    lazy_expr = (lazy_a + lazy_b) * lazy_c - (lazy_a / 2.0)
    print("Intermediate Lazy Expression Tree:")
    print(lazy_expr)

    # Compute compiles the operations and executes them at once
    print("\\nCompiling and Executing kernel...")
    result_lazy = lazy_expr.compute()
    print(f"Lazy Result:  {result_lazy}")

    print("\\n--- 2. Performance Benefits ---")
    print("For large arrays, compiling multiple ops into a single pass")
    print("saves memory bandwidth, giving a significant speedup!")


if __name__ == "__main__":
    main()
