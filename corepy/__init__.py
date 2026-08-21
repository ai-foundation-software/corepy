"""
Corepy: A unified, high-performance core runtime.

Rust-native API with automatic backend selection (CPU/GPU).

Example:
    >>> import corepy as cp
    >>> arr = cp.array([1.0, 2.0, 3.0])
    >>> arr.shape
    (3,)
    >>> z = cp.zeros((3, 3))
    >>> result = arr.sum()
"""

import io
import os
import platform
from typing import Optional, Tuple, Union

# MacOS Metal Library detection
if platform.system() == "Darwin":
    bundled_lib = os.path.join(os.path.dirname(__file__), "default.metallib")
    if os.path.exists(bundled_lib):
        os.environ["COREPY_METAL_LIB_PATH"] = bundled_lib

# Windows DLL Handling for BLAS libraries (OpenBLAS / MKL)
# Python 3.8+ on Windows no longer searches PATH for DLL dependencies.
# We must explicitly register directories or preload the DLL.
if platform.system() == "Windows":
    import sys

    _dll_search_dirs: list = []

    # 1. Check for DLL co-located with the package (CI copies it here)
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(_pkg_dir, "libopenblas.dll")):
        _dll_search_dirs.append(_pkg_dir)

    # 2. MKL DLLs from virtual environment (mkl-devel installs here)
    #    build.rs auto-detects MKL from VIRTUAL_ENV/Library/lib/mkl_rt.lib
    #    The runtime DLLs are in VIRTUAL_ENV/Library/bin/
    for _env_var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        _venv = os.environ.get(_env_var)
        if _venv:
            _mkl_bin = os.path.join(_venv, "Library", "bin")
            if os.path.exists(_mkl_bin):
                _dll_search_dirs.append(_mkl_bin)
    # Also check sys.prefix (uv venvs set this)
    _sys_mkl_bin = os.path.join(sys.prefix, "Library", "bin")
    if os.path.exists(_sys_mkl_bin):
        _dll_search_dirs.append(_sys_mkl_bin)

    # 3. OpenBLAS from OPENBLAS_DIR env var (set by CI or manual install)
    _openblas_env = os.environ.get("OPENBLAS_DIR")
    if _openblas_env:
        _abs_openblas = os.path.abspath(_openblas_env)
        _bin_dir = os.path.join(_abs_openblas, "bin")
        if os.path.exists(_bin_dir):
            _dll_search_dirs.append(_bin_dir)
        if os.path.exists(_abs_openblas):
            _dll_search_dirs.append(_abs_openblas)

    # Register all candidate directories with Windows DLL loader
    for _d in _dll_search_dirs:
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(_d)
        except OSError:
            pass

    # 4. Fallback: preload BLAS DLL via ctypes
    _blas_loaded = False
    for _dll_name in ("mkl_rt.dll", "libopenblas.dll"):
        if _blas_loaded:
            break
        for _d in _dll_search_dirs:
            _candidate = os.path.join(_d, _dll_name)
            if os.path.exists(_candidate):
                try:
                    import ctypes

                    ctypes.WinDLL(_candidate)  # type: ignore[attr-defined]
                    _blas_loaded = True
                    break
                except OSError:
                    pass

    # Cleanup temp vars
    del _dll_search_dirs, _blas_loaded

try:
    from corepy import data, runtime, schema
except ImportError:
    pass

from . import (
    backend,
    buffer_pool,  # Import buffer pool module
    lazy,  # Import lazy evaluation module
)
from .buffer import GPUBuffer

# Primary exports
from .array import _map_to_rust_dtype, ndarray
from .ops import math as _math_ops  # Trigger registration

# UFUNC CORE-12: Import all operation modules
from .ops.arithmetic import (
    add as add,
)
from .ops.arithmetic import (
    divide,
    floor_divide,
    multiply,
    power,
    subtract,
)
from .ops.arithmetic import mod as mod
from .ops.bitwise import (
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    left_shift,
    right_shift,
)
from .ops.comparison import (
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    not_equal,
)
from .ops.creation import empty_like, full_like, identity, ones_like, zeros_like
from .ops.exponential import exp, exp2, expm1, log, log1p, log2, log10, sqrt
from .ops.indexing import boolean_index, take
from .ops.logic import (
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
)
from .ops.math import maximum, minimum
from .ops.reduction import (
    cumprod,
    cumsum,
    max,
    mean,
    min,
    nanmax,
    nanmean,
    nanmin,
    nansum,
    prod,
    std,
    sum,
    var,
)
from .ops.rounding import ceil, clamp, clip, copysign, floor, rint, round_, sign, trunc
from .ops.searching import argmax, argmin, searchsorted
from .ops.searching import where_ as where
from .ops.sorting import argsort, sort, stable_sort
from .ops.special import absolute, cbrt, negative, positive, reciprocal, square

abs = absolute
from .ops.shape import flatten, ravel, reshape, squeeze, transpose
from .ops.stacking import array_split, hstack, repeat, split, stack, tile, vstack

# UFUNC CORE-50: Import all new operation modules
from .ops.trigonometry import (
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    cos,
    cosh,
    degrees,
    hypot,
    radians,
    sin,
    sinh,
    tan,
    tanh,
)
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
from .random import rand, randn

try:
    from . import _corepy_rust  # type: ignore[import-not-found, import-untyped]
except ImportError:
    try:
        import _corepy_rust  # type: ignore[import-not-found, import-untyped]
    except ImportError:
        pass

from . import linalg
from .dataframe import DataFrame, read_csv
from .series import Series

__version__ = "0.3.2"

# Expose types and backend control
from .backend import (
    BackendPolicy,
    DataType,
    analyse_workload,
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

# Type aliases for backward compatibility where generic array interfaces are expected
float32 = DataType.FLOAT32
float64 = DataType.FLOAT64
int32 = DataType.INT32
int64 = DataType.INT64
bool = DataType.BOOL

# Dtype string mapping (pure Python, no NumPy)
_DTYPE_STR_MAP = {
    "float32": DataType.FLOAT32,
    "float64": DataType.FLOAT64,
    "int32": DataType.INT32,
    "int64": DataType.INT64,
    "bool": DataType.BOOL,
}


def _dtype_from_string(dtype_str):
    """Convert dtype string to CorePy DataType."""
    if isinstance(dtype_str, DataType):
        return dtype_str
    return _DTYPE_STR_MAP.get(str(dtype_str), DataType.FLOAT32)


# =============================================================================
# Factory Functions
# =============================================================================


def concatenate(arrays, axis=0) -> ndarray:
    """
    Join a sequence of arrays along axis 0.

    Args:
        arrays: Sequence of arrays (ndarray or list).
        axis: The axis along which to join (only 0 supported).

    Returns:
        ndarray: The concatenated array.
    """
    try:
        from ._corepy_rust import _RustCoreArray as _CT

        core_arrays = []
        first_backend = None
        ndarrays = []
        for arr in arrays:
            if not isinstance(arr, ndarray):
                arr = ndarray(arr)
            ndarrays.append(arr)
            if first_backend is None:
                first_backend = arr.backend

        if axis != 0:
            if axis == 1 and all(len(arr.shape) == 2 for arr in ndarrays):
                transposed = [arr.transpose() for arr in ndarrays]
                return concatenate(transposed, axis=0).transpose()
            else:
                raise NotImplementedError(
                    f"Concatenation along axis {axis} is not yet supported."
                )

        for arr in ndarrays:
            if arr._core_array is not None:
                core_arrays.append(arr._core_array)
            else:
                flat = list(_flatten_data(arr.to_list()))
                ct = _CT(flat, list(arr._shape))
                core_arrays.append(ct)

        result_ct = _CT.concatenate(core_arrays)
        result_arr = ndarray(
            [],
            dtype=DataType.FLOAT32,
            device=None,
            backend=first_backend,
        )
        result_arr._shape = tuple(result_ct.shape)
        result_arr._element_count = result_ct.element_count
        result_arr._core_array = result_ct
        return result_arr
    except ImportError:
        combined = []
        first_backend = None
        for arr in arrays:
            if isinstance(arr, ndarray):
                if first_backend is None:
                    first_backend = arr.backend
                combined.extend(_flatten_data(arr._core_array.to_list()))
            elif isinstance(arr, (list, tuple)):
                combined.extend(_flatten_data(arr))
        return ndarray(combined, dtype=DataType.FLOAT32, backend=first_backend)


def _flatten_data(items):
    """Flatten nested lists/tuples into a flat iterator."""
    if isinstance(items, (list, tuple)):
        for x in items:
            yield from _flatten_data(x)
    else:
        yield float(items)


def array(
    data, dtype: DataType = DataType.FLOAT32, device: Optional[str] = None
) -> ndarray:
    """
    Create an array from data.

    This is the primary way to create arrays in CorePy.
    Automatically switches between eager and lazy evaluation based on context.

    Args:
        data: Input data (list, tuple, or existing ndarray).
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
    """Create an array filled with zeros."""
    if isinstance(shape, int):
        shape = (shape,)
    from ._corepy_rust import _RustCoreArray as _CT

    rust_dtype = _map_to_rust_dtype(dtype)
    ct = _CT.zeros(list(shape), rust_dtype)
    return ndarray._wrap_core_array(ct, dtype, device or "cpu")


def ones(
    shape: Union[int, Tuple[int, ...]],
    dtype: DataType = DataType.FLOAT32,
    device: Optional[str] = None,
) -> ndarray:
    """Create an array filled with ones."""
    if isinstance(shape, int):
        shape = (shape,)
    from ._corepy_rust import _RustCoreArray as _CT

    ct = _CT.ones(list(shape))
    return ndarray._wrap_core_array(ct, dtype, device or "cpu")


def empty(
    shape: Union[int, Tuple[int, ...]],
    dtype: DataType = DataType.FLOAT32,
    device: Optional[str] = None,
) -> ndarray:
    """Create an uninitialized array (mapped to zeros for safety)."""
    return zeros(shape, dtype, device)


def full(
    shape: Union[int, Tuple[int, ...]],
    fill_value: float,
    dtype: DataType = DataType.FLOAT32,
    device: Optional[str] = None,
) -> ndarray:
    """Create an array filled with a specific value."""
    if isinstance(shape, int):
        shape = (shape,)
    from ._corepy_rust import _RustCoreArray as _CT

    ct = _CT.full(list(shape), float(fill_value))
    return ndarray._wrap_core_array(ct, dtype, device or "cpu")


def eye(
    n: int,
    m: Optional[int] = None,
    dtype: DataType = DataType.FLOAT32,
    device: Optional[str] = None,
) -> ndarray:
    """Create an identity matrix."""
    from ._corepy_rust import _RustCoreArray as _CT

    ct = _CT.eye(n, m)
    return ndarray._wrap_core_array(ct, dtype, device or "cpu")


def stack(
    arrays: Union[list, tuple],
    axis: int = 0,
    device: Optional[str] = None,
) -> ndarray:
    """Stack arrays along a new axis."""
    if not arrays:
        raise ValueError("Need at least one array to stack")
    from .array import ndarray

    arr_objs = [a if isinstance(a, ndarray) else ndarray(a) for a in arrays]
    try:
        from ._corepy_rust import _RustCoreArray as _CT

        core_arrays = [a._ensure_core_array() for a in arr_objs]
        if all(ca is not None for ca in core_arrays):
            ct = _CT.stack(core_arrays, axis)
            return ndarray._wrap_core_array(ct, DataType.FLOAT32, device or "cpu")
    except Exception:
        pass
    from .ops.stacking import stack as _stack

    return _stack(arr_objs, axis=axis)


def split(
    array: Union[ndarray, list],
    indices_or_sections: Union[int, list],
    axis: int = 0,
) -> list:
    """Split an array into multiple sub-arrays."""
    from .ops.stacking import split as _split

    return _split(array, indices_or_sections, axis=axis)


def squeeze(
    array: ndarray,
    axis: Optional[int] = None,
) -> ndarray:
    """Remove single-dimensional entries from the shape of an array."""
    return array.squeeze(axis)


def arange(
    start,
    stop=None,
    step=1,
    dtype: DataType = DataType.FLOAT32,
    device: Optional[str] = None,
) -> ndarray:
    """
    Create an array with evenly spaced values.

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
    from ._corepy_rust import _RustCoreArray as _CT

    if stop is None:
        stop = start
        start = 0
    ct = _CT.arange(float(start), float(stop), float(step))
    return ndarray._wrap_core_array(ct, dtype, device or "cpu")


def linspace(
    start,
    stop,
    num: int = 50,
    dtype: DataType = DataType.FLOAT32,
    device: Optional[str] = None,
) -> ndarray:
    """
    Return evenly spaced numbers over [start, stop].

    Args:
        start: Start value.
        stop: Stop value.
        num: Number of samples (default: 50).
        dtype: Data type.
        device: Target device.

    Returns:
        ndarray: Array of evenly spaced values.

    Example:
    """
    if num == 0:
        return ndarray([], dtype=dtype, device=device)
    if num == 1:
        return ndarray([float(start)], dtype=dtype, device=device)
    try:
        from ._corepy_rust import _RustCoreArray as _CT

        ct = _CT.linspace(float(start), float(stop), int(num))
        return ndarray._wrap_core_array(ct, dtype, device or "cpu")
    except ImportError:
        step = (float(stop) - float(start)) / (num - 1)
        data = [float(start) + step * i for i in range(num)]
        return ndarray(data, dtype=dtype, device=device)
        step = (float(stop) - float(start)) / (num - 1)
        data = [float(start) + step * i for i in range(num)]
        return ndarray(data, dtype=dtype, device=device)


def is_even(a) -> ndarray:
    """
    Element-wise is_even detection.

    Args:
        a: Input array or list.

    Returns:
        ndarray with 1.0 where element is even, 0.0 otherwise.

    Example:
        >>> cp.is_even([1, 2, 3, 4])
        ndarray([0., 1., 0., 1.])
    """
    from .ops.ufunc_engine import ufunc_unary

    return ufunc_unary("is_even", a)


def is_odd(a) -> ndarray:
    """
    Element-wise is_odd detection.

    Args:
        a: Input array or list.

    Returns:
        ndarray with 1.0 where element is odd, 0.0 otherwise.

    Example:
        >>> cp.is_odd([1, 2, 3, 4])
        ndarray([1., 0., 1., 0.])
    """
    from .ops.ufunc_engine import ufunc_unary

    return ufunc_unary("is_odd", a)


def _add_compat(a, b) -> ndarray:
    """
    Element-wise addition of two arrays.

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
    Matrix multiplication.

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


# Alias for dot product
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


# Deprecated alias (kept for backward compatibility) - REMOVED

# Import lazy context manager for export
from typing import Any

from .lazy import lazy  # Export context manager: with cp.lazy()

__all__ = [
    # Core types
    "ndarray",
    "DataType",
    # Factory functions
    "array",
    "zeros",
    "ones",
    "full",
    "empty",
    "arange",
    "linspace",
    "rand",
    "randn",
    "concatenate",
    "DataFrame",
    "Series",
    "linalg",
    # Lazy evaluation
    "lazy",
    # Profiling
    "ProfileContext",
    "clear_profile",
    "enable_profiling",
    "disable_profiling",
    "detect_bottlenecks",
    "detect_regressions",
    "profile_operation",
    "profile_report",
    "export_profile",
    "get_recommendations",
    "compute_stats",
    # Data types
    "Float32",
    "Float64",
    "Int32",
    "Int64",
    "Bool",
    # Aliases
    "float32",
    "float64",
    "int32",
    "int64",
    "bool",
    # Backend control
    "BackendPolicy",
    "get_backend_policy",
    "set_backend_policy",
    "explain_last_dispatch",
    "analyse_workload",
    "get_device_info",
    # UFUNC CORE-12: Arithmetic
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "mod",
    "floor_divide",
    # UFUNC CORE-12: Comparison
    "equal",
    "not_equal",
    "greater",
    "less",
    "greater_equal",
    "less_equal",
    # UFUNC CORE-12: Logical
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    # UFUNC CORE-12: Sorting
    "sort",
    "argsort",
    "stable_sort",
    # UFUNC CORE-12: Searching
    "argmax",
    "argmin",
    "where",
    "searchsorted",
    "minimum",
    "maximum",
    # UFUNC CORE-12: Indexing
    "take",
    "boolean_index",
    # UFUNC CORE-12: Utilities
    "is_even",
    "is_odd",
    "abs",
    # Math
    "matmul",
    "dot",
    # UFUNC CORE-50: Trigonometric
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    # UFUNC CORE-50: Hyperbolic
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    # UFUNC CORE-50: Angle conversion
    "degrees",
    "radians",
    "hypot",
    # UFUNC CORE-50: Exponential / Logarithmic
    "exp",
    "exp2",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "sqrt",
    # UFUNC CORE-50: Rounding / Sign / Clip
    "floor",
    "ceil",
    "round_",
    "trunc",
    "rint",
    "sign",
    "clip",
    "clamp",
    "copysign",
    # UFUNC CORE-50: Bitwise
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "left_shift",
    "right_shift",
    # UFUNC CORE-50: Reductions
    "prod",
    "std",
    "var",
    "cumsum",
    "cumprod",
    "nansum",
    "nanmean",
    "nanmax",
    "nanmin",
    # UFUNC CORE-50: Special
    "square",
    "reciprocal",
    "cbrt",
    "positive",
    "negative",
    "absolute",
    # UFUNC CORE-50: Creation
    "zeros_like",
    "ones_like",
    "full_like",
    "empty_like",
    "eye",
    "identity",
    # UFUNC CORE-50: Stacking
    "stack",
    "vstack",
    "hstack",
    "split",
    "array_split",
    "tile",
    "repeat",
    "add_one",
]


def add_one(val: int) -> int:
    """Legacy helper function for testing."""
    return val + 1
