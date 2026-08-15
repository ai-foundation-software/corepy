"""
UFUNC CORE-12: Logical Operations

Provides NumPy-compatible functional API for logical operations:
    cp.logical_and(a, b)
    cp.logical_or(a, b)
    cp.logical_not(a)
    cp.logical_xor(a, b)

Operates on float representation: 0.0 = False, non-zero = True.
Returns arrays with 1.0 (true) and 0.0 (false).
"""

from typing import Any

from .ufunc_engine import ufunc_binary, ufunc_unary


def logical_and(a: Any, b: Any) -> Any:
    """Element-wise logical AND."""
    return ufunc_binary("logical_and", a, b)


def logical_or(a: Any, b: Any) -> Any:
    """Element-wise logical OR."""
    return ufunc_binary("logical_or", a, b)


def logical_not(a: Any) -> Any:
    """Element-wise logical NOT (unary)."""
    return ufunc_unary("logical_not", a)


def logical_xor(a: Any, b: Any) -> Any:
    """Element-wise logical XOR."""
    return ufunc_binary("logical_xor", a, b)
