from typing import Any, List

from ..backend.dispatch import register_kernel
from ..backend.types import BackendType

# ============================================================================
# ARCHITECTURE COMPLIANCE NOTE:
# ============================================================================
# This file registers kernel implementations for the dispatch system.
#
# According to the 3-layer architecture (execution_model.md):
# - Python Layer: Zero performance logic (INTENT ONLY)
# - Rust Layer: Validation, scheduling, memory management
# - C++ Layer: Pure execution (SIMD kernels)
#
# When Rust/C++ FFI is available, the ndarray class uses FFI directly.
# These Python fallbacks are used ONLY when the Rust extension is not
# compiled, providing cross-platform compatibility (ARM/x86).
#
# Performance Note: These fallbacks are significantly slower than FFI.
# For production use, compile the Rust extension with `maturin develop`.
# ============================================================================


def _flatten(data: Any) -> List[float]:
    """Recursively flatten nested lists/tuples/arrays to a flat list of floats."""
    import numpy as np

    # Handle numpy arrays first
    if isinstance(data, np.ndarray):
        return [float(x) for x in data.flatten()]

    # Handle memoryview
    if isinstance(data, memoryview):
        return [float(x) for x in data.tolist()]

    # Handle bytes/bytearray
    if isinstance(data, (bytes, bytearray)):
        return [float(x) for x in data]

    # Handle lists/tuples recursively
    if isinstance(data, (list, tuple)):
        result = []
        for item in data:
            result.extend(_flatten(item))
        return result

    # Scalar value
    return [float(data)]


def _reshape(flat: List[float], shape: tuple) -> Any:
    """Reshape a flat list back to nested structure based on shape."""
    if len(shape) == 0:
        return flat[0] if flat else 0.0
    if len(shape) == 1:
        return flat

    # Multi-dimensional reshape
    size = 1
    for dim in shape[1:]:
        size *= dim

    return [
        _reshape(flat[i * size : (i + 1) * size], shape[1:]) for i in range(shape[0])
    ]


@register_kernel("add", BackendType.CPU)
def cpu_add(a: Any, b: Any) -> Any:
    """
    Element-wise addition (Python fallback).

    Used when Rust FFI is not available. Slower but cross-platform compatible.
    """
    flat_a = _flatten(a)
    flat_b = _flatten(b)

    if len(flat_a) != len(flat_b):
        raise ValueError(f"Shape mismatch: {len(flat_a)} vs {len(flat_b)}")

    return [x + y for x, y in zip(flat_a, flat_b)]


@register_kernel("sub", BackendType.CPU)
def cpu_sub(a: Any, b: Any) -> Any:
    """
    Element-wise subtraction (Python fallback).
    """
    flat_a = _flatten(a)
    flat_b = _flatten(b)

    if len(flat_a) != len(flat_b):
        raise ValueError(f"Shape mismatch: {len(flat_a)} vs {len(flat_b)}")

    return [x - y for x, y in zip(flat_a, flat_b)]


@register_kernel("mul", BackendType.CPU)
def cpu_mul(a: Any, b: Any) -> Any:
    """
    Element-wise multiplication (Python fallback).
    """
    flat_a = _flatten(a)
    flat_b = _flatten(b)

    if len(flat_a) != len(flat_b):
        raise ValueError(f"Shape mismatch: {len(flat_a)} vs {len(flat_b)}")

    return [x * y for x, y in zip(flat_a, flat_b)]


@register_kernel("div", BackendType.CPU)
def cpu_div(a: Any, b: Any) -> Any:
    """
    Element-wise division (Python fallback).
    """
    flat_a = _flatten(a)
    flat_b = _flatten(b)

    if len(flat_a) != len(flat_b):
        raise ValueError(f"Shape mismatch: {len(flat_a)} vs {len(flat_b)}")

    return [x / y for x, y in zip(flat_a, flat_b)]


@register_kernel("sum", BackendType.CPU)
def cpu_sum(a: Any) -> float:
    """
    Sum reduction (Python fallback).
    """
    flat = _flatten(a)
    return sum(flat)


@register_kernel("mean", BackendType.CPU)
def cpu_mean(a: Any) -> float:
    """
    Mean reduction (Python fallback).
    """
    flat = _flatten(a)
    if len(flat) == 0:
        return 0.0
    return sum(flat) / len(flat)


@register_kernel("matmul", BackendType.CPU)
def cpu_matmul(a: Any, b: Any, shape_a: Any = None, shape_b: Any = None) -> Any:
    """
    Matrix multiplication (Python fallback).

    Supports:
    - 1D @ 1D: dot product -> scalar
    - 2D @ 2D: matrix multiply -> 2D
    """
    flat_a = _flatten(a)
    flat_b = _flatten(b)

    # 2D case
    if shape_a and len(shape_a) == 2 and shape_b and len(shape_b) == 2:
        m, k1 = shape_a
        k2, n = shape_b

        if k1 != k2:
            raise ValueError(f"Matrix dimension mismatch: k={k1} vs k={k2}")

        # Result M x N
        # Naive O(M*N*K) implementation
        result_rows = []
        for i in range(m):
            row = []
            for j in range(n):
                val = 0.0
                for k in range(k1):
                    val += flat_a[i * k1 + k] * flat_b[k * n + j]
                row.append(val)
            result_rows.append(row)
        return result_rows

    # Fallback/1D case
    if len(flat_a) != len(flat_b):
        raise ValueError(f"Dot product size mismatch: {len(flat_a)} vs {len(flat_b)}")

    return sum(x * y for x, y in zip(flat_a, flat_b))


@register_kernel("all", BackendType.CPU)
def cpu_all(a: Any) -> bool:
    """
    Reduction: all() operation (Python fallback).
    """
    flat = _flatten(a)
    return all(bool(x) for x in flat)


@register_kernel("any", BackendType.CPU)
def cpu_any(a: Any) -> bool:
    """
    Reduction: any() operation (Python fallback).
    """
    flat = _flatten(a)
    return any(bool(x) for x in flat)


@register_kernel("std", BackendType.CPU)
def cpu_std(a: Any) -> float:
    """
    Standard deviation (Python fallback).
    """
    flat = _flatten(a)
    if len(flat) == 0:
        return 0.0
    mean = sum(flat) / len(flat)
    variance = sum((x - mean) ** 2 for x in flat) / len(flat)
    return variance**0.5


@register_kernel("max", BackendType.CPU)
def cpu_max(a: Any) -> float:
    """
    Max reduction (Python fallback).
    """
    flat = _flatten(a)
    if len(flat) == 0:
        return 0.0
    return max(flat)


@register_kernel("min", BackendType.CPU)
def cpu_min(a: Any) -> float:
    """
    Min reduction (Python fallback).
    """
    flat = _flatten(a)
    if len(flat) == 0:
        return 0.0
    return min(flat)
