"""
UFUNC CORE-12 Benchmark: CorePy vs NumPy

Compares performance across:
- Element-wise arithmetic (add, multiply, power)
- Sorting
- Reductions (argmax)
- Various array sizes (100 → 1M elements)
"""

import time

import corepy as cp


def benchmark(name, fn, warmup=3, repeats=20):
    """Run a benchmark and return median time in ms."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times.sort()
    median = times[len(times) // 2]
    return median


def _bench_binary(name, arr_a, arr_b, op):
    """Benchmark a binary operation, binding arrays via default args."""
    return benchmark(name, lambda a=arr_a, b=arr_b: op(a, b))


def _bench_unary(name, arr, op):
    """Benchmark a unary operation, binding array via default arg."""
    return benchmark(name, lambda a=arr: op(a))


def main():
    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]

    print("=" * 70)
    print("UFUNC CORE-50 Benchmark")
    print("=" * 70)

    # Try import numpy
    try:
        import numpy as np

        has_numpy = True
    except ImportError:
        has_numpy = False
        print("NumPy not available — CorePy-only benchmarks\n")

    for size in sizes:
        print(f"\n--- Array Size: {size:,} ---")

        # Create arrays
        cp_a = cp.array([float(i) for i in range(size)])
        cp_b = cp.array([float(i + 1) for i in range(size)])

        if has_numpy:
            np_a = np.arange(size, dtype=np.float32)
            np_b = np.arange(1, size + 1, dtype=np.float32)

        # ADD
        cp_add_ms = _bench_binary("cp.add", cp_a, cp_b, lambda a, b: a + b)
        line = f"  add:      CorePy={cp_add_ms:.3f}ms"
        if has_numpy:
            np_add_ms = _bench_binary("np.add", np_a, np_b, lambda a, b: a + b)
            ratio = np_add_ms / cp_add_ms if cp_add_ms > 0 else float("inf")
            line += f"  NumPy={np_add_ms:.3f}ms  ratio={ratio:.2f}x"
        print(line)

        # MULTIPLY
        cp_mul_ms = _bench_binary("cp.mul", cp_a, cp_b, lambda a, b: a * b)
        line = f"  multiply: CorePy={cp_mul_ms:.3f}ms"
        if has_numpy:
            np_mul_ms = _bench_binary("np.mul", np_a, np_b, lambda a, b: a * b)
            ratio = np_mul_ms / cp_mul_ms if cp_mul_ms > 0 else float("inf")
            line += f"  NumPy={np_mul_ms:.3f}ms  ratio={ratio:.2f}x"
        print(line)

        # POWER
        cp_pow_ms = _bench_unary("cp.pow", cp_a, lambda a: a**2)
        line = f"  power:    CorePy={cp_pow_ms:.3f}ms"
        if has_numpy:
            np_pow_ms = _bench_unary("np.pow", np_a, lambda a: a**2)
            ratio = np_pow_ms / cp_pow_ms if cp_pow_ms > 0 else float("inf")
            line += f"  NumPy={np_pow_ms:.3f}ms  ratio={ratio:.2f}x"
        print(line)

        # SORT (only up to 100K to avoid long waits)
        if size <= 100_000:
            cp_sort_ms = _bench_unary("cp.sort", cp_a, lambda a: a.sort())
            line = f"  sort:     CorePy={cp_sort_ms:.3f}ms"
            if has_numpy:
                np_sort_ms = _bench_unary("np.sort", np_a, lambda a: np.sort(a))
                ratio = np_sort_ms / cp_sort_ms if cp_sort_ms > 0 else float("inf")
                line += f"  NumPy={np_sort_ms:.3f}ms  ratio={ratio:.2f}x"
            print(line)

        # ARGMAX
        cp_argmax_ms = _bench_unary("cp.argmax", cp_a, lambda a: a.argmax())
        line = f"  argmax:   CorePy={cp_argmax_ms:.3f}ms"
        if has_numpy:
            np_argmax_ms = _bench_unary("np.argmax", np_a, lambda a: np.argmax(a))
            ratio = np_argmax_ms / cp_argmax_ms if cp_argmax_ms > 0 else float("inf")
            line += f"  NumPy={np_argmax_ms:.3f}ms  ratio={ratio:.2f}x"
        print(line)

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
