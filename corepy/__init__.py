"""
Corepy: A unified, high-performance core runtime.
"""

import os
import platform

# MacOS Metal Library detection
if platform.system() == "Darwin":
    # If the wheel bundled the metallib, it should be here
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
                os.add_dll_directory(bin_dir)
            except Exception:
                pass # Fallback, might already be in PATH

from corepy import data, runtime, schema

from . import backend
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
from .tensor import Tensor

try:
    from ._corepy_cpp import add_one  # type: ignore[import-untyped]
except ImportError:
    # Fallback or warning if extension is not present (e.g. during dev without compile)
    def add_one(x: int) -> int:
        raise ImportError("C++ extension not loaded. Did you install with -v?")


try:
    from . import _corepy_rust  # type: ignore[import-untyped]
except ImportError:
    try:
        import _corepy_rust  # type: ignore[import-untyped]
    except ImportError:
        pass  # Managed in usage sites

tensor = Tensor

__version__ = "0.2.3"

# Expose types and backend control
from .backend import (
    BackendPolicy,
    DataType,
    detect_devices,
    explain_last_dispatch,
    get_backend_policy,
    set_backend_policy,
)

get_device_info = detect_devices


def matmul(a, b):
    """
    Matrix multiplication (dot product or 2D matmul).
    Delegates to Tensor.matmul.
    """
    if not isinstance(a, Tensor):
        a = Tensor(a)
    if not isinstance(b, Tensor):
        b = Tensor(b)
    return a.matmul(b)


Float32 = DataType.FLOAT32
Float64 = DataType.FLOAT64
Int32 = DataType.INT32
Int64 = DataType.INT64
Bool = DataType.BOOL


def compute_stats(tensor: Tensor, stats: list) -> dict:
    """
    Compute multiple statistics on a tensor in one call.

    Args:
        tensor: Input tensor
        stats: List of stat names to compute (e.g., ["mean", "sum", "std"])

    Returns:
        Dictionary mapping stat names to their computed values
    """
    result = {}
    for stat in stats:
        if stat == "mean":
            result[stat] = tensor.mean()
        elif stat == "sum":
            result[stat] = tensor.sum()
        elif stat == "std":
            result[stat] = tensor.std()
        elif stat == "max":
            result[stat] = tensor.max()
        elif stat == "min":
            result[stat] = tensor.min()
        else:
            raise ValueError(f"Unknown stat: {stat}")
    return result


__all__ = [
    "data",
    "schema",
    "runtime",
    "add_one",
    "Tensor",
    "tensor",
    "backend",
    "profiler",
    "enable_profiling",
    "disable_profiling",
    "clear_profile",
    "profile_report",
    "export_profile",
    "ProfileContext",
    "profile_operation",
    "detect_bottlenecks",
    "get_recommendations",
    "detect_regressions",
    "compute_stats",
    "Float32",
    "Float64",
    "Int32",
    "Int64",
    "Bool",
    "DataType",
    "BackendPolicy",
    "get_backend_policy",
    "set_backend_policy",
    "explain_last_dispatch",
    "get_device_info",
    "matmul",
    "dot",
]

# Alias dor dot product
dot = matmul
