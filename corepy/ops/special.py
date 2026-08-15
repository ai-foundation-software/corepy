"""
UFUNC CORE-50: Special Mathematical Functions

    cp.square, cp.reciprocal, cp.cbrt, cp.positive, cp.negative, cp.absolute
"""

import math
from typing import Any

from .trigonometry import _unary_trig


def square(a: Any) -> Any:
    """Element-wise square: x²."""
    return _unary_trig("square", "square", lambda x: x * x, a)


def reciprocal(a: Any) -> Any:
    """Element-wise reciprocal: 1/x."""
    return _unary_trig("reciprocal", "reciprocal", lambda x: 1.0 / x, a)


def cbrt(a: Any) -> Any:
    """Element-wise cube root."""
    return _unary_trig("cbrt", "cbrt", lambda x: math.copysign(abs(x) ** (1 / 3), x), a)


def positive(a: Any) -> Any:
    """Element-wise positive (identity for numeric)."""
    return _unary_trig("positive", "clone", lambda x: +x, a)


def negative(a: Any) -> Any:
    """Element-wise negative: -x."""
    return _unary_trig("neg", "neg", lambda x: -x, a)


def absolute(a: Any) -> Any:
    """Element-wise absolute value."""
    return _unary_trig("abs", "abs", abs, a)
