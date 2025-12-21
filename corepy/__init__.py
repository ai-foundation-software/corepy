"""
Corepy: A unified, high-performance core runtime.
"""
from corepy import data, schema, runtime
from .tensor import Tensor
from . import backend
from .ops import math as _math_ops # Trigger registration

try:
    from ._corepy_cpp import add_one
except ImportError:
    # Fallback or warning if extension is not present (e.g. during dev without compile)
    def add_one(x: int) -> int:
        raise ImportError("C++ extension not loaded. Did you install with -v?")

__version__ = "0.1.0"

__all__ = ["data", "schema", "runtime", "add_one", "Tensor", "backend"]
