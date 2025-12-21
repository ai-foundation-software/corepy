# CPU/GPU Backend Selection Architecture

Corepy implements a **Correctness-First** backend selection strategy. This ensures that operations only run on accelerators (GPU/TPU) when it is safe, performant, and guaranteed to be correct.

## Core Principles

1.  **Correctness > Speed**: If an operation's safety on GPU is ambiguous, it runs on CPU.
2.  **CPU Default**: The system defaults to CPU. GPU is an opt-in optimization for specific workloads.
3.  **Data Scalability**: Small data stays on CPU to avoid transfer overhead.
4.  **Explicitness**: Users can always force a specific backend via API or Environment Variables.

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

The `select_backend` function determines the execution device:

```python
def select_backend(op, data, device_info):
    # 1. User/Env Override
    if forced: return forced
    
    # 2. Safety Checks
    if op.type is CONTROL: return CPU
    
    # 3. Optimality Checks
    if has_gpu and data.size > threshold:
        return GPU
        
    return CPU
```

## User Overrides

### Environment Variable
Set `COREPY_BACKEND` to force a default for the entire process:

```bash
export COREPY_BACKEND=gpu  # Try to run everything on GPU (unsafe fallback to CPU if missing)
export COREPY_BACKEND=cpu  # Force strict CPU execution
```

### API Override
Pass `device` or `backend` argument to tensor constructors or operations:

```python
import corepy.backend as cb

# Manual override
backend = cb.select_backend(..., requested_backend=cb.BackendType.GPU)
```

## Cross-Platform Notes

*   **Linux**: Supports CUDA (Nvidia) and ROCm (AMD).
*   **macOS**: Supports Metal (MPS) on Apple Silicon.
*   **Windows**: Supports CUDA.

*Note: Initial implementation uses placeholders for driver detection. Full hardware integration will follow.*
