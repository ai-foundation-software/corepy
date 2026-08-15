"""
UFUNC CORE-12: Arithmetic Operations

Provides NumPy-compatible functional API for arithmetic:
    cp.add(a, b)
    cp.subtract(a, b)
    cp.multiply(a, b)
    cp.divide(a, b)
    cp.power(a, b)
    cp.mod(a, b)
    cp.floor_divide(a, b)

All support multi-input: cp.add(a, b, c, d)
"""

from typing import Any

from .ufunc_engine import ufunc_binary, ufunc_multi


def add(a: Any, b: Any, *more: Any) -> Any:
    """Element-wise addition. Supports multi-input: add(a, b, c, d)."""
    if more:
        return ufunc_multi("add", a, b, *more)
    return ufunc_binary("add", a, b)


def subtract(a: Any, b: Any, *more: Any) -> Any:
    """Element-wise subtraction. Supports multi-input."""
    if more:
        return ufunc_multi("sub", a, b, *more)
    return ufunc_binary("sub", a, b)


def multiply(a: Any, b: Any, *more: Any) -> Any:
    """Element-wise multiplication. Supports multi-input."""
    if more:
        return ufunc_multi("mul", a, b, *more)
    return ufunc_binary("mul", a, b)


def divide(a: Any, b: Any) -> Any:
    """Element-wise division."""
    return ufunc_binary("div", a, b)


def power(a: Any, b: Any) -> Any:
    """Element-wise power: a ** b."""
    return ufunc_binary("power", a, b)


def mod(a: Any, b: Any) -> Any:
    """Element-wise modulo: a % b."""
    return ufunc_binary("mod", a, b)


def floor_divide(a: Any, b: Any) -> Any:
    """Element-wise floor division: a // b."""
    return ufunc_binary("floor_div", a, b)
