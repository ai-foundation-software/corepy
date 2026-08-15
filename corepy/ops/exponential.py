"""
UFUNC CORE-50: Exponential & Logarithmic Operations

    cp.exp, cp.exp2, cp.expm1, cp.log, cp.log2, cp.log10, cp.log1p
"""

import math
from typing import Any

from .trigonometry import _unary_trig


def exp(a: Any) -> Any:
    """Element-wise exponential: e^x."""
    return _unary_trig("exp", "exp", math.exp, a)


def exp2(a: Any) -> Any:
    """Element-wise 2^x."""
    return _unary_trig("exp2", "exp2", lambda x: 2.0**x, a)


def expm1(a: Any) -> Any:
    """Element-wise exp(x) - 1 (accurate for small x)."""
    return _unary_trig("expm1", "expm1", math.expm1, a)


def log(a: Any) -> Any:
    """Element-wise natural logarithm."""
    return _unary_trig("log", "log", math.log, a)


def log2(a: Any) -> Any:
    """Element-wise base-2 logarithm."""
    return _unary_trig("log2", "log2", math.log2, a)


def log10(a: Any) -> Any:
    """Element-wise base-10 logarithm."""
    return _unary_trig("log10", "log10", math.log10, a)


def log1p(a: Any) -> Any:
    """Element-wise log(1 + x) (accurate for small x)."""
    return _unary_trig("log1p", "log1p", math.log1p, a)


def sqrt(a: Any) -> Any:
    """Element-wise square root."""
    return _unary_trig("sqrt", "sqrt", math.sqrt, a)
