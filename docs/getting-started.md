# Getting Started with Corepy

This guide will walk you through installing Corepy and writing your first high-performance array program.

## Prerequisites

- **Python**: 3.10+
- **OS**: Linux, macOS, or Windows
- **Arch**: x86_64, aarch64 (Apple Silicon)
- **Features**: AVX2 (Intel/AMD), Metal (macOS)


## 1. Installation

Corepy is available on PyPI.

```bash
pip install corepy-ai
```

### Manual Installation
If installing from source, navigate to the project root and run:
```bash
pip install .
```

### Verification
To verify your installation contains all necessary backend components:
```bash
python -c "import corepy; print(f'Corepy v{corepy.__version__} installed successfully.')"
```

## 2. Your First Program

Corepy works with `Array` objects. Unlike Python lists, Arrays are strictly typed and store data in contiguous memory (managed by Rust/C++), allowing for extremely fast operations.

Create a file named `hello_corepy.py`:

```python
import corepy as cp


def main():
    # 1. Hardware Analysis
    print("Backend Analysis:")
    print(cp.analyse_workload(matrix_size=1024))
    print("-" * 30)

    # 2. Array Math
    prices = cp.array([10.5, 20.0, 15.5, 30.0])
    total = prices.sum()
    average = prices.mean()

    print(f"Prices: {prices}\\nTotal: {total}\\nAverage: {average}\\n")

    # 3. High-Performance Random Matrix
    # Fast multi-threaded generation straight from Rust
    rand_matrix = cp.random.randn((1000, 1000), seed=42)
    print(f"Random Matrix Mean: {rand_matrix.mean()}")


if __name__ == "__main__":
    main()
```

Run it:
```bash
python hello_corepy.py
```

Expected Output:
```
Prices: Array([10.5, 20.0, 15.5, 30.0], backend='cpu')
Total:   Array([76.0], backend='cpu')
Average: Array([19.0], backend='cpu')
```

## 3. Key Differences from NumPy

If you are coming from NumPy, here are the main things to watch out for:

### Explicit Backends
Corepy is "device-aware". While it often picks the best default, it exposes the backend explicitly.

```python
# Check where your array lives
print(prices.backend)  # e.g., BackendType.CPU or BackendType.GPU
```

### Strict Types
Corepy enforces data types more strictly to prevent silent overflows or precision loss during high-performance compute kernels.

```python
import corepy as cp

# Defaults to Float32 for performance, not Float64!
t = cp.array([1, 2, 3])
```

### Immutable by Default (mostly)
Many operations return *new* Arrays rather than modifying in-place, enabling safer parallelism.

## Next Steps

## Next Steps

- Learn about the [Core Concepts](02_core_concepts/core-concepts.md) (Backends, DataFrames, Random Ops).
- See [Architecture](03_architecture/architecture.md) to understand the Rust engine setup.
- Try the Tutorials under `tutorials/`, specifically:
  - `06_advanced_features.py`: Hardware caps and OS profiling
  - `07_dataframe_api.py`: DataFrames, GroupBy, Pivots, and CSV I/O
  - `08_random_numbers.py`: Multi-threaded robust Random Number Generation
  - `09_lazy_evaluation.py`: Operation Fusion and Lazy Evaluation
  - `10_memory_management.py`: Memory allocations and custom Buffer Pools
