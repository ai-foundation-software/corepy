"""
UFUNC CORE-12: Universal Function Engine

Core dispatch layer for elementwise operations with:
- Broadcasting support
- Multi-input chaining
- Backend dispatch (CoreArray → Python fallback)
- Scalar coercion
"""

from typing import Any, Callable, Optional, Tuple, Union

from ..broadcasting import broadcast_shapes


def _ensure_array(x: Any) -> Any:
    """Convert scalars and lists to ndarray if needed and ensure CoreArray is ready."""
    from ..array import ndarray

    if not isinstance(x, ndarray):
        x = ndarray(x if isinstance(x, (list, tuple)) else [x])
    x._ensure_core_array()
    return x


def _broadcast_pair(a: Any, b: Any) -> tuple:
    """
    Broadcast two arrays to compatible shapes.
    Returns (a_broadcasted, b_broadcasted).
    """
    from ..array import ndarray

    a = _ensure_array(a)
    b = _ensure_array(b)

    if a.shape == b.shape:
        return a, b

    # Compute target shape
    target_shape = broadcast_shapes(a.shape, b.shape)

    # 1. Expand a to target_shape
    if a.shape != target_shape:
        if a.size == 1:
            scalar = a.to_list()[0] if hasattr(a, "to_list") else float(a)
            total = 1
            for d in target_shape:
                total *= d
            a = ndarray([scalar] * total).reshape(target_shape)
        else:
            # Multi-dimensional broadcasting (physical expansion)
            # This is a slow fallback but ensures correctness for now
            # TODO: Move this to Rust for high performance
            a = _broadcast_to(a, target_shape)

    # 2. Expand b to target_shape
    if b.shape != target_shape:
        if b.size == 1:
            scalar = b.to_list()[0] if hasattr(b, "to_list") else float(b)
            total = 1
            for d in target_shape:
                total *= d
            b = ndarray([scalar] * total).reshape(target_shape)
        else:
            b = _broadcast_to(b, target_shape)

    return a, b


def _broadcast_to(arr: Any, target_shape: Tuple[int, ...]) -> Any:
    """Physically broadcast an array to a target shape by repeating data."""
    from ..array import ndarray

    if arr.shape == target_shape:
        return arr

    # Right-align shapes
    ndim_orig = len(arr.shape)
    ndim_target = len(target_shape)

    # Simple case: (3,) to (3, 3) or (1, 3) to (3, 3)
    # We'll use a recursive approach to expand dimensions from right to left
    data = arr.to_list()

    # Flatten and repeat logic (simplified for common cases)
    # For a full implementation, we'd use strides or a more complex repeat pattern.
    # Here we'll do a simple nested list expansion then re-wrap.

    def expand_recursive(current_data, current_shape, target_shape):
        if not target_shape:
            return current_data

        if not current_shape:
            # We have a scalar/item, but target expects a dimension
            # Repeat this item for the first dimension of target
            inner = expand_recursive(current_data, (), target_shape[1:])
            return [inner] * target_shape[0]

        if current_shape[0] == target_shape[0]:
            # Dimensions match, go deeper
            return [
                expand_recursive(item, current_shape[1:], target_shape[1:])
                for item in current_data
            ]
        elif current_shape[0] == 1:
            # Dimension is 1, broadcast to target_shape[0]
            inner = expand_recursive(
                current_data[0], current_shape[1:], target_shape[1:]
            )
            return [inner] * target_shape[0]
        else:
            # This should have been caught by broadcast_shapes
            raise ValueError(f"Incompatible shapes: {current_shape} vs {target_shape}")

    # Align original shape with target by prepending 1s
    aligned_shape = (1,) * (ndim_target - ndim_orig) + arr.shape

    # Reshape internal list to have the prepended 1s
    def wrap_ones(data, n):
        for _ in range(n):
            data = [data]
        return data

    aligned_data = wrap_ones(data, ndim_target - ndim_orig)
    expanded_data = expand_recursive(aligned_data, aligned_shape, target_shape)

    return ndarray(expanded_data, dtype=arr.dtype, backend=arr.backend)


def ufunc_binary(op_name: str, a: Any, b: Any) -> Any:
    """
    Execute a binary ufunc with broadcasting.

    Args:
        op_name: Operation name ('add', 'sub', 'mul', 'div', 'power', 'mod', 'floor_div',
                 'eq', 'ne', 'gt', 'lt', 'ge', 'le',
                 'logical_and', 'logical_or', 'logical_xor')
        a: First operand (array or scalar)
        b: Second operand (array or scalar)

    Returns:
        ndarray with the result
    """
    from ..array import ndarray

    a, b = _broadcast_pair(a, b)

    import time

    from ..profiler.core import record_op

    start_time = time.perf_counter()

    # Try CoreArray fast path
    if a._core_array is not None and b._core_array is not None:
        core_ops = {
            "add": a._core_array.add,
            "sub": a._core_array.sub,
            "mul": a._core_array.mul,
            "div": a._core_array.div,
            "power": a._core_array.power,
            "mod": a._core_array.mod_op,
            "floor_div": a._core_array.floor_div,
            "eq": a._core_array.eq,
            "ne": a._core_array.ne,
            "gt": a._core_array.gt,
            "lt": a._core_array.lt,
            "ge": a._core_array.ge,
            "le": a._core_array.le,
            "logical_and": a._core_array.logical_and,
            "logical_or": a._core_array.logical_or,
            "logical_xor": a._core_array.logical_xor,
            "maximum": a._core_array.maximum,
            "minimum": a._core_array.minimum,
        }
        core_fn = core_ops.get(op_name)
        if core_fn is not None:
            result_ca = core_fn(b._core_array)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op(op_name, elapsed_ms, "Rust-CPU")
            return ndarray._from_core_array(result_ca, dtype=a.dtype, backend=a.backend)

    # Python fallback for basic arithmetic
    from .math import _flatten

    flat_a = _flatten(a._core_array.to_list() if a._core_array is not None else a)
    flat_b = _flatten(b._core_array.to_list() if b._core_array is not None else b)

    py_ops: dict[str, Callable] = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "power": lambda x, y: x**y,
        "mod": lambda x, y: x % y,
        "floor_div": lambda x, y: x // y,
        "eq": lambda x, y: 1.0 if x == y else 0.0,
        "ne": lambda x, y: 1.0 if x != y else 0.0,
        "gt": lambda x, y: 1.0 if x > y else 0.0,
        "lt": lambda x, y: 1.0 if x < y else 0.0,
        "ge": lambda x, y: 1.0 if x >= y else 0.0,
        "le": lambda x, y: 1.0 if x <= y else 0.0,
        "logical_and": lambda x, y: 1.0 if (x != 0 and y != 0) else 0.0,
        "logical_or": lambda x, y: 1.0 if (x != 0 or y != 0) else 0.0,
        "logical_xor": lambda x, y: 1.0 if (bool(x) ^ bool(y)) else 0.0,
        "maximum": lambda x, y: max(x, y),
        "minimum": lambda x, y: min(x, y),
    }

    fn = py_ops.get(op_name)
    if fn is None:
        raise ValueError(f"Unknown binary ufunc: {op_name}")

    result_data = [fn(x, y) for x, y in zip(flat_a, flat_b)]
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    record_op(op_name, elapsed_ms, "Python-Fallback")
    result = ndarray(result_data, dtype=a.dtype, backend=a.backend)
    result._shape = a.shape
    return result


def ufunc_unary(op_name: str, a: Any) -> Any:
    """Execute a unary ufunc."""
    import math
    import time

    from ..array import ndarray
    from ..profiler.core import record_op

    start_time = time.perf_counter()
    a = _ensure_array(a)

    # CoreArray fast path
    if a._core_array is not None:
        core_ops = {
            "logical_not": a._core_array.logical_not,
            "neg": a._core_array.neg,
            "abs": a._core_array.abs,
            "is_even": a._core_array.is_even,
            "is_odd": a._core_array.is_odd,
        }
        core_fn = core_ops.get(op_name)
        if core_fn is not None:
            result_ca = core_fn()
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            record_op(op_name, elapsed_ms, "Rust-CPU")
            return ndarray._from_core_array(result_ca, dtype=a.dtype, backend=a.backend)

    # Python fallback
    from .math import _flatten

    flat = _flatten(a._core_array.to_list() if a._core_array is not None else a)

    py_ops: dict[str, Callable] = {
        "logical_not": lambda x: 1.0 if x == 0 else 0.0,
        "neg": lambda x: -x,
        "abs": lambda x: abs(x),
        "is_even": lambda x: 1.0 if x % 2 == 0 else 0.0,
        "is_odd": lambda x: 1.0 if x % 2 != 0 else 0.0,
    }

    fn = py_ops.get(op_name)
    if fn is None:
        raise ValueError(f"Unknown unary ufunc: {op_name}")

    result_data = [fn(x) for x in flat]
    result = ndarray(result_data, dtype=a.dtype, backend=a.backend)
    result._shape = a.shape
    return result


def ufunc_multi(op_name: str, *arrays: Any) -> Any:
    """
    Execute a binary ufunc across multiple inputs via left-fold.

    Example: ufunc_multi('add', a, b, c, d) → ((a + b) + c) + d
    """
    if len(arrays) < 2:
        raise ValueError(f"ufunc '{op_name}' requires at least 2 inputs")

    result = ufunc_binary(op_name, arrays[0], arrays[1])
    for arr in arrays[2:]:
        result = ufunc_binary(op_name, result, arr)
    return result
