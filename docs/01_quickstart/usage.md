# 📖 Usage Guide

Corepy is designed to be familiar if you have used tools like NumPy or Pandas, but with added safety and speed.

> [!NOTE]
> **Version 0.2.0 Status**: This guide shows what's currently working (✅) and what's planned for future releases (🔮).

---

## ✅ Working Features (v0.2.0)

These examples work in the current released version and have been tested.

---

### 🌟 Example 1: Basic Tensor Operations (Math & Reductions)

Corepy supports standard arithmetic and reduction operations.

```python
import corepy as cp

# Create tensors with list data
a = cp.Tensor([1.0, 2.0, 3.0])
b = cp.Tensor([4.0, 5.0, 6.0])

# Element-wise arithmetic (Works!)
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")

# Scalar operations
print(f"a * 2.0 = {a * 2.0}")

# Reductions
print(f"Sum: {a.sum()}")
print(f"Mean: {a.mean()}")
```

**What works:**
- ✅ Tensor creation from Python lists
- ✅ Element-wise arithmetic (`+`, `-`, `*`, `/`)
- ✅ Scalar operations (`tensor * scalar`)
- ✅ Reductions (`sum`, `mean`)
- ✅ CPU backend (automatic)

**What doesn't work yet:**
- ❌ Matrix multiplication (`@`) - partially implemented but not exposed via operator

---

### 🚀 Example 2: Performance Features (Arena & Parallelism)

Corepy v0.2.2 introduces advanced memory management and parallel execution.

```python
# Automatic Parallelism
# Tensors > 1M elements automatically use multi-threaded execution
large_tensor = cp.Tensor([1.0] * 1_000_000)
result = large_tensor.sum() # Runs in parallel using Rayon
```

**New Features:**
- ✅ **Arena Allocation**: Zero-overhead memory management for reductions
- ✅ **Parallel Dispatch**: Automatic multi-threading for large arrays
- ✅ **SIMD**: AVX2 optimized kernels (where available)

---

### 🔍 Example 3: Device Detection

Corepy can detect your system's hardware capabilities automatically.

```python
from corepy.backend import detect_devices

# Get comprehensive system information
info = detect_devices()

print(f"CPU Cores: {info.cpu_cores}")
print(f"Has AVX2: {info.has_avx2}")
print(f"Platform: {info.platform_system}")
```

**Example Output (Linux x86_64)**:
```
CPU Cores: 4
Has AVX2: True
Has AVX-512: False
Has NEON: False
GPU Count: 0
Platform: Linux
```

**Example Output (Apple M1)**:
```
CPU Cores: 8
Has AVX2: False
Has AVX-512: False
Has NEON: True
GPU Count: 1
Platform: Darwin
```

**Use Cases:**
- Understand what hardware optimizations are available
- Debug platform-specific issues

---

### 📊 Example 4: Data Tables

Basic data table functionality for structured data.

```python
from corepy.data import Table

# Create a table from a dictionary
data = {
    "name": ["Alice", "Bob", "Charlie"],
    "score": [95.5, 87.3, 92.1],
    "age": [25, 30, 28]
}

table = Table(data)
print(table)
# Output: Table(rows=3, schema=None)
```

**Status**: Basic functionality works. Schema integration and advanced operations coming in future versions.

---

### 🧪 Example 5: Testing & Debugging

Verify your installation and explore the backend.

```python
import corepy as cp
from corepy.backend import ReferenceBackend

# Check version
print(f"Corepy version: {cp.__version__}")

# Access reference backend (pure Python implementation for debugging)
ref_backend = ReferenceBackend()
```

---

## 🧠 Core Concepts (v1)

Understanding Corepy's design philosophy:

1. **Correctness First**: Corepy prioritizes getting the right answer over raw speed. Operations fail fast rather than silently producing incorrect results.

2. **Eager Execution**: v1 uses straightforward, eager execution—what you write is what executes immediately (lazy graphs planned for v2).

3. **CPU-Optimized**: v1 focuses exclusively on CPU performance using C++ SIMD optimizations (GPU support planned for v2).

4. **Backend Abstraction**: The architecture is designed for future multi-device support, even though v1 is CPU-only.

5. **Type Safety**: Strong type hints and schema integration prevent common data pipeline errors.

---

## 📚 Additional Resources

- **[Platform Support Guide](platform_support.md)**: CPU & GPU setup for Linux, macOS, and Windows
- **[Backend Selection Guide](../02_core_concepts/backend_selection.md)**: Backend architecture and device selection
- **[Installation Guide](install.md)**: Detailed setup instructions
- **[Contributing Guide](../07_contributing/CONTRIBUTING.md)**: How to contribute to Corepy

---

## 🔮 Coming Soon (Future Versions)

The following features are planned but not yet implemented. These examples show the envisioned API.

---

### 🖼️ Future Example: Image Processing Pipeline

**Status**: 🔮 Planned for v2  
**Dependencies**: I/O module, Vision module, Lazy execution

```python
import corepy as cp

def preprocess_images(folder_path):
    """
    AI image preprocessing pipeline.
    NOTE: This API is not implemented yet.
    """
    # 1. Find all images
    files = cp.io.glob(f"{folder_path}/*.jpg")  # ❌ Not implemented
    
    # 2. Load images in parallel
    images_batch = cp.io.read_images(files)  # ❌ Not implemented
    
    # 3. Normalize pixel values
    normalized = (images_batch - 0.485) / 0.229  # ❌ Operator not implemented
    
    # 4. Resize
    resized = cp.vision.resize(normalized, (224, 224))  # ❌ Module not implemented
    
    # 5. Execute on best hardware
    return resized.compute(device="auto")  # ❌ Lazy execution not implemented

# This will work in v2.0+
# batch = preprocess_images("./my_data")
```

**Planned for**: v2.0 (6-12 months)

---

### 📈 Future Example: Financial Data Processing

**Status**: 🔮 Planned for v1.5  
**Dependencies**: IPC reader, Advanced aggregations, Windowing

```python
import corepy.data as cpd
import corepy as cp

def process_financial_data():
    """
    High-frequency trading data analysis.
    NOTE: This API is not fully implemented yet.
    """
    # Read Arrow IPC format
    df = cpd.read_ipc("trade_data.arrow")  # ❌ Not implemented
    
    # Calculate rolling statistics
    stats = (
        df.select("symbol", "price")
        .group_by("symbol")
        .rolling(window="1s")  # ❌ Not implemented
        .agg(
            avg_price=cp.mean("price"),  # ❌ Not implemented
            volatility=cp.std("price")   # ❌ Not implemented
        )
    )
    
    return stats

# This will work in v1.5+
# results = process_financial_data()
```

**Planned for**: v1.5 (3-6 months)

---

### ⚡ Future Example: Advanced Tensor Operations

**Status**: 🔮 Planned for v0.3  
**Dependencies**: Complete tensor operations

```python
import corepy as cp

# These operations are planned but not yet implemented
a = cp.Tensor([1.0, 2.0, 3.0])
b = cp.Tensor([4.0, 5.0, 6.0])

# Coming in v0.3 (1-2 months)
c = a - b          # ❌ Subtraction
d = a * b          # ❌ Element-wise multiplication
e = a / b          # ❌ Division
f = a * 2.0        # ❌ Scalar multiplication

# Matrix operations
A = cp.Tensor([[1, 2], [3, 4]])
B = cp.Tensor([[5, 6], [7, 8]])
C = A @ B          # ❌ Matrix multiplication

# Reductions
sum_val = a.sum()  # ❌ Not implemented
mean_val = a.mean()  # ❌ Not implemented
```

**Planned for**: v0.3 (1-2 months)

---

### 🎯 Future Example: GPU Acceleration

**Status**: 🔮 Planned for v2.0  
**Dependencies**: GPU backend, Device transfer

```python
import corepy as cp

# GPU support coming in v2.0
tensor_cpu = cp.Tensor([1, 2, 3], device="cpu")

# Move to GPU
tensor_gpu = tensor_cpu.to("cuda:0")  # ❌ Not implemented

# Operations automatically run on GPU
result = tensor_gpu * 2.0  # ❌ Not implemented

# Move back to CPU
result_cpu = result.to("cpu")  # ❌ Not implemented
```

**Planned for**: v2.0 (6-12 months)

---

### 🧪 Future Example: Reference Backend for Testing

**Status**: 🔮 Exists but not exposed in v0.2  
**Dependencies**: Expose ReferenceBackend in public API

```python
import corepy as cp
from corepy.backend import ReferenceBackend  # ❌ Not exposed yet

# Test C++ implementation against pure Python
a = [[1.0, 2.0], [3.0, 4.0]]
b = [[5.0, 6.0], [7.0, 8.0]]

# C++ optimized path
result_cpp = cp.matmul(a, b)  # ❌ Not implemented

# Pure Python reference (slow but correct)
result_ref = ReferenceBackend.matmul(a, b)  # ❌ Not exposed

# Should always match
assert result_cpp == result_ref
```

**Planned for**: v0.3 (exposed in public API)

---

### 🚀 Future Example: Lazy Execution and Optimization

**Status**: 🔮 Planned for v2.0  
**Dependencies**: Execution graph, Graph optimizer

```python
import corepy as cp

# Create computation graph (lazy)
x = cp.Tensor([1, 2, 3])
y = cp.Tensor([4, 5, 6])

# These don't execute immediately - they build a graph
z = x + y
w = z * 2
result = w.sum()

# Execution happens here, with optimizations
final = result.compute()  # ❌ Lazy execution not implemented

# Corepy will optimize:
# - Fuse operations (add + multiply + sum in one kernel)
# - Choose best device (CPU vs GPU)
# - Minimize memory allocations
```

**Planned for**: v2.0 (lazy execution graph system)

---

## 🚫 Current Limitations

**v0.2.2 Work in Progress:**

### Operations
- ❌ Matrix multiplication (`@`) operator not yet hooked up
- ❌ Advanced linear algebra (inverse, svd, etc.)

### Modules
- ❌ No I/O module (`cp.io`)
- ❌ No vision module (`cp.vision`)

### Execution
- ❌ GPU support (Planned for v2.0)
- ❌ Lazy execution graph (Planned for v2.0)

---

## 💡 Best Practices (Current Version)

### ✅ Do This

```python
# Use tensor addition (works)
a = cp.Tensor([1, 2, 3])
b = cp.Tensor([4, 5, 6])
result = a + b

# Check device capabilities
info = detect_devices()
if info.has_avx2:
    print("AVX2 available for future optimizations")

# Use Data.Table for structured data
from corepy.data import Table
table = Table({"col1": [1, 2], "col2": [3, 4]})
```

### ❌ Don't Do This (Yet)

```python
# These don't work in v0.2.2
result = a @ b         # ❌ Matrix multiplication (operator not hooked up)
result = cp.io.read()  # ❌ Module doesn't exist
```

---

## 🔄 Version History

| Version | Status | Key Features |
|:--------|:-------|:-------------|
| **v0.2.0** | ✅ Current | Tensor creation, addition, device detection, basic tables |
| **v0.3.0** | 🔨 In Progress | All tensor operations, C++ extension in wheel, reference backend |
| **v1.0** | 🔮 Planned | CPU SIMD optimizations, multi-threading, schema system |
| **v2.0** | 🔮 Planned | GPU support, lazy execution, I/O and vision modules |

---

## ❓ FAQ

**Q: Why do only some operations work?**  
A: Corepy is in early alpha (v0.2.0). We're implementing features incrementally, prioritizing correctness over completeness.

**Q: When will GPU support be available?**  
A: GPU backend is planned for v2.0, approximately 6-12 months from now.

**Q: How can I help?**  
A: See [CONTRIBUTING.md](../07_contributing/CONTRIBUTING.md) for ways to contribute. Implementing missing tensor operations is a great starting point!

**Q: Is the C++ extension working?**  
A: The C++ code exists, but may not be included in the PyPI wheel for v0.2.0. We're fixing this in v0.3.

**Q: Should I use Corepy in production?**  
A: Not yet. v0.2.0 is for experimentation and feedback. Wait for v1.0 for production use.

---

**Last Updated**: 2026-02-05
**Next Planned Release**: v0.3.0 (March 2026)  
**Documentation Version**: Matches released package v0.2.2
