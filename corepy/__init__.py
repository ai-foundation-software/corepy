"""
Corepy: A unified, high-performance core runtime.

NumPy-compatible API with automatic backend selection (CPU/GPU).

Example:
    >>> import corepy as cp
    >>> arr = cp.array([1.0, 2.0, 3.0])
    >>> arr.shape
    (3,)
    >>> z = cp.zeros((3, 3))
    >>> result = arr.sum()
"""

import os
import platform
from typing import Optional, Tuple, Union

import numpy as np

# MacOS Metal Library detection
if platform.system() == "Darwin":
    bundled_lib = os.path.join(os.path.dirname(__file__), "default.metallib")
    if os.path.exists(bundled_lib):
        os.environ["COREPY_METAL_LIB_PATH"] = bundled_lib

# Windows DLL Handling for OpenBLAS
if platform.system() == "Windows":
    openblas_dir = os.environ.get("OPENBLAS_DIR")
    if openblas_dir:
        bin_dir = os.path.join(openblas_dir, "bin")
        if os.path.exists(bin_dir):
            try:
                if hasattr(os, "add_dll_directory"):  # Windows only
                    os.add_dll_directory(bin_dir)  # type: ignore[attr-defined]
            except Exception:
                pass

from corepy import data, runtime, schema

from . import (
    backend,
    buffer_pool,  # Import buffer pool module
    lazy,  # Import lazy evaluation module
)

# NumPy-compatible primary exports
from .array import Tensor, ndarray  # ndarray is primary, Tensor is deprecated alias
from .ops import math as _math_ops  # Trigger registration
from .profiler import (
    ProfileContext,
    clear_profile,
    detect_bottlenecks,
    detect_regressions,
    disable_profiling,
    enable_profiling,
    export_profile,
    get_recommendations,
    profile_operation,
    profile_report,
)

try:
    from ._corepy_cpp import add_one  # type: ignore[import-untyped]
except ImportError:

    def add_one(x: int) -> int:
        raise ImportError("C++ extension not loaded. Did you install with -v?")


try:
    from . import _corepy_rust  # type: ignore[import-not-found, import-untyped]
except ImportError:
    try:
        import _corepy_rust  # type: ignore[import-not-found, import-untyped]
    except ImportError:
        pass

__version__ = "0.2.4"

# Expose types and backend control
from .backend import (
    BackendPolicy,
    DataType,
    detect_devices,
    explain_last_dispatch,
    get_backend_policy,
    get_system_capabilities,
    set_backend_policy,
)

get_device_info = detect_devices

# Dtype shortcuts
Float32 = DataType.FLOAT32
Float64 = DataType.FLOAT64
Int32 = DataType.INT32
Int64 = DataType.INT64
Bool = DataType.BOOL

# NumPy-compatible aliases
float32 = DataType.FLOAT32
float64 = DataType.FLOAT64
int32 = DataType.INT32
int64 = DataType.INT64
bool = DataType.BOOL

# Dtype conversion helper
_DTYPE_TO_NUMPY = {
    DataType.FLOAT32: np.float32,
    DataType.FLOAT64: np.float64,
    DataType.INT32: np.int32,
    DataType.INT64: np.int64,
    DataType.BOOL: bool,
}


def _dtype_to_numpy(dtype: DataType):
    """Convert CorePy dtype to NumPy dtype."""
    return _DTYPE_TO_NUMPY.get(dtype, np.float32)


# Reverse mapping for inference
_NUMPY_TO_DTYPE = {
    np.dtype("float32"): DataType.FLOAT32,
    np.dtype("float64"): DataType.FLOAT64,
    np.dtype("int32"): DataType.INT32,
    np.dtype("int64"): DataType.INT64,
    np.dtype("bool"): DataType.BOOL,
    # String aliases
    np.float32: DataType.FLOAT32,
    np.float64: DataType.FLOAT64,
    np.int32: DataType.INT32,
    np.int64: DataType.INT64,
    bool: DataType.BOOL,
}


def _dtype_from_numpy(np_dtype):
    """Convert NumPy dtype to CorePy dtype."""
    # Handle numpy dtype objects
    if isinstance(np_dtype, np.dtype):
        # Try exact match first
        if np_dtype in _NUMPY_TO_DTYPE:
            return _NUMPY_TO_DTYPE[np_dtype]
        # Try by type
        return _NUMPY_TO_DTYPE.get(np_dtype.type, DataType.FLOAT32)
    return _NUMPY_TO_DTYPE.get(np_dtype, DataType.FLOAT32)


# =============================================================================
# NumPy-Compatible Factory Functions
# =============================================================================


def concatenate(arrays, axis=0) -> ndarray:
    """
    Join a sequence of arrays along an existing axis (NumPy-compatible).

    Args:
        arrays: Sequence of arrays (ndarray, numpy array, or list).
        axis: The axis along which the arrays will be joined.

    Returns:
        ndarray: The concatenated array.
    """
    # Convert all inputs to numpy arrays first (simplest implementation)
    np_arrays = []
    first_backend = None

    for arr in arrays:
        if isinstance(arr, ndarray):
            np_arrays.append(arr.to_numpy())
            if first_backend is None:
                first_backend = arr.backend
        elif isinstance(arr, (list, tuple)):
            np_arrays.append(np.array(arr))
        elif isinstance(arr, np.ndarray):
            np_arrays.append(arr)
        else:
            raise ValueError(f"Unsupported type for concatenation: {type(arr)}")

    result = np.concatenate(np_arrays, axis=axis)

    # Infer CorePy dtype from result
    cp_dtype = _dtype_from_numpy(result.dtype)

    return ndarray(result, dtype=cp_dtype, device=None, backend=first_backend)


def array(
    data, dtype: DataType = DataType.FLOAT32, device: Optional[str] = None
) -> ndarray:
    """
    Create an array from data (NumPy-compatible).

    This is the primary way to create arrays in CorePy, matching np.array().
    Automatically switches between eager and lazy evaluation based on context.

    Args:
        data: Input data (list, tuple, numpy array, or existing ndarray).
        dtype: Data type (default: FLOAT32).
        device: Target device ('cpu', 'metal', 'gpu').

    Returns:
        ndarray in normal mode, LazyArray in lazy mode

    Example:
        >>> import corepy as cp
        >>> # Eager mode
        >>> arr = cp.array([1.0, 2.0, 3.0])
        >>> result = arr + arr  # Executes immediately
        >>>
        >>> # Lazy mode (same API!)
        >>> with cp.lazy():
        ...     arr = cp.array([1.0, 2.0, 3.0])
        ...     result = arr + arr  # Builds expression tree
        ...     materialized = result.compute()
    """
    from .lazy import LazyArray
    from .lazy.context import is_lazy_mode

    arr = ndarray(data, dtype=dtype, device=device)

    # Wrap in LazyArray if in lazy mode
    if is_lazy_mode():
        return LazyArray(arr)  # type: ignore[return-value]

    return arr


def zeros(
    shape: Union[int, Tuple[int, ...]],
    dtype: DataType = DataType.FLOAT32,
    device: Optional[str] = None,
) -> ndarray:
    """
    Create an array filled with zeros (NumPy-compatible).

    Args:
        shape: Shape of the array as int or tuple.
        dtype: Data type (default: FLOAT32).
        device: Target device.

    Returns:
        ndarray: Array of zeros.

    Example:
        >>> cp.zeros((2, 3))
        ndarray([[0., 0., 0.], [0., 0., 0.]])
    """
    if isinstance(shape, int):
        shape = (shape,)
    np_arr = np.zeros(shape, dtype=_dtype_to_numpy(dtype))
    return ndarray(np_arr, dtype=dtype, device=device)


def ones(
    shape: Union[int, Tuple[int, ...]],
    dtype: DataType = DataType.FLOAT32,
    device: Optional[str] = None,
) -> ndarray:
    """
    Create an array filled with ones (NumPy-compatible).

    Args:
        shape: Shape of the array as int or tuple.
        dtype: Data type (default: FLOAT32).
        device: Target device.

    Returns:
        ndarray: Array of ones.

    Example:
        >>> cp.ones((2, 2))
        ndarray([[1., 1.], [1., 1.]])
    """
    if isinstance(shape, int):
        shape = (shape,)
    np_arr = np.ones(shape, dtype=_dtype_to_numpy(dtype))
    return ndarray(np_arr, dtype=dtype, device=device)


def empty(
    shape: Union[int, Tuple[int, ...]],
    dtype: DataType = DataType.FLOAT32,
    device: Optional[str] = None,
) -> ndarray:
    """
    Create an uninitialized array (NumPy-compatible).

    Args:
        shape: Shape of the array as int or tuple.
        dtype: Data type (default: FLOAT32).
        device: Target device.

    Returns:
        ndarray: Uninitialized array (contents are undefined).

    Example:
        >>> e = cp.empty((3,))  # Fast allocation, values undefined
    """
    if isinstance(shape, int):
        shape = (shape,)
    np_arr = np.empty(shape, dtype=_dtype_to_numpy(dtype))
    return ndarray(np_arr, dtype=dtype, device=device)


def arange(
    start,
    stop=None,
    step=1,
    dtype: DataType = DataType.FLOAT32,
    device: Optional[str] = None,
) -> ndarray:
    """
    Create an array with evenly spaced values (NumPy-compatible).

    Args:
        start: Start value (or stop if stop is None).
        stop: Stop value (exclusive).
        step: Step size.
        dtype: Data type (default: FLOAT32).
        device: Target device.

    Returns:
        ndarray: Array with values [start, start+step, start+2*step, ...).

    Example:
        >>> cp.arange(0, 10, 2)
        ndarray([0., 2., 4., 6., 8.])
    """
    np_arr = np.arange(start, stop, step, dtype=_dtype_to_numpy(dtype))
    return ndarray(np_arr, dtype=dtype, device=device)


def add(a, b) -> ndarray:
    """
    Element-wise addition of two arrays (NumPy-compatible).

    Args:
        a: First array or scalar.
        b: Second array or scalar.

    Returns:
        ndarray: Result of a + b.

    Example:
        >>> cp.add([1, 2], [3, 4])
        ndarray([4., 6.])
    """
    if not isinstance(a, ndarray):
        a = ndarray(a)
    if not isinstance(b, ndarray):
        b = ndarray(b)
    return a + b


def matmul(a, b) -> ndarray:
    """
    Matrix multiplication (NumPy-compatible).

    Computes dot product for 1D arrays, matrix multiplication for 2D.

    Args:
        a: First array.
        b: Second array.

    Returns:
        ndarray: Result of a @ b.

    Example:
        >>> a = cp.array([[1, 2], [3, 4]])
        >>> b = cp.array([[5, 6], [7, 8]])
        >>> cp.matmul(a, b)
        ndarray([[19., 22.], [43., 50.]])
    """
    if not isinstance(a, ndarray):
        a = ndarray(a)
    if not isinstance(b, ndarray):
        b = ndarray(b)
    return a.matmul(b)


# Alias for dot product (NumPy compatibility)
dot = matmul


def compute_stats(arr: ndarray, stats: list) -> dict:
    """
    Compute multiple statistics on an array in one call.

    Args:
        arr: Input array.
        stats: List of stat names (e.g., ["mean", "sum", "std"]).

    Returns:
        Dictionary mapping stat names to computed values.
    """
    result = {}
    for stat in stats:
        if stat == "mean":
            result[stat] = arr.mean()
        elif stat == "sum":
            result[stat] = arr.sum()
        elif stat == "std":
            result[stat] = arr.std()
        elif stat == "max":
            result[stat] = arr.max()
        elif stat == "min":
            result[stat] = arr.min()
        else:
            raise ValueError(f"Unknown stat: {stat}")
    return result


# Deprecated alias (kept for backward compatibility)
tensor = Tensor

# Import lazy context manager for export
from .lazy import lazy  # Export context manager: with cp.lazy()

# NumPy-compatible type hints
NDArray = np.ndarray

# BackendType for explicit backend selection
__all__ = [
    # Core types
    "ndarray",
    "Tensor",
    "DataType",
    "BackendType",
    "NDArray",
    # Factory functions
    "array",
    "zeros",
    "ones",
    "arange",
    "linspace",
    # Lazy evaluation
    "lazy",
    # Profiling
    "ProfileContext",
    "clear_profile",
    "enable_profiling",
    "disable_profiling",
    "record_op_time",
    "detect_bottlenecks",
    "detect_regressions",
    "add_one",
    "compute_stats",
    # Data types
    "Float32",
    "Float64",
    "Int32",
    "Int64",
    "Bool",
    # NumPy aliases
    "float32",
    "float64",
    "int32",
    "int64",
    "bool",
    "DataType",
    # Backend control
    "BackendPolicy",
    "get_backend_policy",
    "set_backend_policy",
    "explain_last_dispatch",
    "get_device_info",
]
