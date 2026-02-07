# Getting Started with Corepy

This guide will walk you through installing Corepy and writing your first high-performance tensor program.

## Prerequisites

Corepy is a hybrid Python/Rust library. 
Corepy is a hybrid Python/Rust library.
- **Python**: 3.9+
- **OS**: Linux, macOS, or Windows
- **Arch**: x86_64 or ARM64 (Apple Silicon)

## 1. Installation

Corepy is available on PyPI.

```bash
pip install corepy
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

Corepy works with `Tensor` objects. Unlike Python lists, Tensors are strictly typed and store data in contiguous memory (managed by Rust/C++), allowing for extremely fast operations.

Create a file named `hello_corepy.py`:

```python
import corepy as cp

def main():
    # 1. Create Data
    # Allocates memory aligned for SIMD operations
    prices = cp.array([10.5, 20.0, 15.5, 30.0])
    
    # 2. Perform Usage
    # This happens at C++ speed (AVX2 optimized)
    total = prices.sum()
    average = prices.mean()
    
    # 3. Output
    print(f"Prices: {prices}")
    print(f"Total:   {total}")
    print(f"Average: {average}")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python hello_corepy.py
```

Expected Output:
```
Prices: Tensor([10.5, 20.0, 15.5, 30.0], backend='cpu')
Total:   Tensor([76.0], backend='cpu')
Average: Tensor([19.0], backend='cpu')
```

## 3. Key Differences from NumPy

If you are coming from NumPy, here are the main things to watch out for:

### Explicit Backends
Corepy is "device-aware". While it often picks the best default, it exposes the backend explicitly.

```python
# Check where your tensor lives
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
Many operations return *new* Tensors rather than modifying in-place, enabling safer parallelism.

## Next Steps

- Learn about the [Core Concepts](core-concepts.md) (Backends, Profiling).
- See [Architecture](architecture.md) to understand the Rust engine.
