# CPU/GPU Backend Selection Architecture

Corepy implements a **Correctness-First** backend selection strategy. This ensures that operations only run on accelerators (GPU/TPU) when it is safe, performant, and guaranteed to be correct.

## Core Principles

1.  **Correctness > Speed**: If an operation's safety on GPU is ambiguous, it runs on CPU.
2.  **CPU Default**: The system defaults to CPU. GPU is an opt-in optimization for specific workloads.
3.  **Rust Mediation**: All execution paths are validated by the Rust runtime for safety before dispatch.
4.  **Data Scalability**: Small data stays on CPU to avoid transfer overhead.
5.  **Explicitness**: Users can always force a specific backend via API or Environment Variables.

## Operation Classification

Every operation in Corepy is classified by `OperationType`:

*   **CONTROL**: Control flow, boolean logic, scalar comparisons. -> **Always CPU**.
*   **MEMORY_BOUND**: Element-wise casts, copies, simple arithmetic. -> **CPU** (unless huge).
*   **COMPUTE_VECTOR**: Heavy vector math (sin, cos, exp, reduction). -> **GPU** (if size > threshold).
*   **COMPUTE_MATRIX**: Matrix multiplication, convolution, decomposition. -> **GPU** (if size > threshold).
*   **SCALAR**: Single value operations. -> **Always CPU**.

## Cost Model & Thresholds

We use conservative thresholds to prevent performance degradation from kernel launch latencies.

| Operation Type | Threshold Condition | Target Backend | Reason |
| :--- | :--- | :--- | :--- |
| **Vector** | Elements > 100,000 | GPU | Amortizes transfer cost. |
| **Matrix** | Shape >= 512x512 | GPU | Compute density outweighs overhead. |
| **Batch** | Batch Size >= 32 | GPU | Sufficient parallelism. |
| **Streaming** | Not Batched | **CPU** | Latency sensitive, avoid PCI-e bottleneck. |

## Backend Selection Logic

The modern Rust-based dispatcher automatically analyzes your hardware and workload sizes:

```python
import corepy as cp

# 1. Automatic Hardware & Cache detection
caps = cp.get_system_capabilities()
print(caps["cpu"])
# Outputs features like `has_avx512`, `has_neon`, and Cache sizes like `l2_cache` for chunking

# 2. Smart dispatch in action
a = cp.ones((2048, 2048))
b = cp.ones((2048, 2048))
result = a @ b  # Rust optimizer decides CPU (Faer/AVX2) or GPU (Metal)
```

## User Overrides

### Environment Variable
Set `COREPY_BACKEND` to force a default for the entire process:

```bash
export COREPY_BACKEND=gpu  # Force GPU usage
export COREPY_BACKEND=cpu  # Force strict CPU execution
```

### API Override
Pass the `backend` argument to constructors:

```python
import corepy as cp

# Explicitly use CPU
t = cp.array([1, 2, 3], backend="cpu")

# Explicitly use GPU (macOS only for now)
try:
    t_gpu = cp.array([1, 2, 3], backend="gpu")
except ValueError:
    print("GPU not available")
```

## Cross-Platform Notes

*   **Linux**: Optimized CPU kernels (Rust AVX2/AVX512, Faer). CUDA support coming soon.
*   **macOS**: Native Metal acceleration on Apple Silicon via Rust.
*   **Windows**: Optimized CPU kernels via Faer/OpenBLAS.

*Note: The system prioritizes Rust native capabilities while aggressively matching L1/L2 Cache topologies to matrix operations for dense performance without GPU overhead.*

