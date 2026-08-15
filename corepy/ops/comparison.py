"""
UFUNC CORE-12: Comparison Operations

Provides NumPy-compatible functional API for comparisons:
    cp.equal(a, b)
    cp.not_equal(a, b)
    cp.greater(a, b)
    cp.less(a, b)
    cp.greater_equal(a, b)
    cp.less_equal(a, b)

Returns arrays with 1.0 (true) and 0.0 (false).
"""

from typing import Any

from .ufunc_engine import ufunc_binary


def equal(a: Any, b: Any) -> Any:
    """Element-wise equality: a == b."""
    return ufunc_binary("eq", a, b)


def not_equal(a: Any, b: Any) -> Any:
    """Element-wise not-equal: a != b."""
    return ufunc_binary("ne", a, b)


def greater(a: Any, b: Any) -> Any:
    """Element-wise greater-than: a > b."""
    return ufunc_binary("gt", a, b)


def less(a: Any, b: Any) -> Any:
    """Element-wise less-than: a < b."""
    return ufunc_binary("lt", a, b)


def greater_equal(a: Any, b: Any) -> Any:
    """Element-wise greater-or-equal: a >= b."""
    return ufunc_binary("ge", a, b)


def less_equal(a: Any, b: Any) -> Any:
    """Element-wise less-or-equal: a <= b."""
    return ufunc_binary("le", a, b)
