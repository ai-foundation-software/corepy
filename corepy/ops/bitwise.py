"""
UFUNC CORE-50: Bitwise Operations

    cp.bitwise_and, cp.bitwise_or, cp.bitwise_xor, cp.bitwise_not
    cp.left_shift, cp.right_shift
"""

from typing import Any

from .trigonometry import _binary_trig, _unary_trig


def bitwise_and(a: Any, b: Any) -> Any:
    """Element-wise bitwise AND (f32 → i32 internally)."""
    return _binary_trig("bitwise_and", lambda x, y: float(int(x) & int(y)), a, b)


def bitwise_or(a: Any, b: Any) -> Any:
    """Element-wise bitwise OR."""
    return _binary_trig("bitwise_or", lambda x, y: float(int(x) | int(y)), a, b)


def bitwise_xor(a: Any, b: Any) -> Any:
    """Element-wise bitwise XOR."""
    return _binary_trig("bitwise_xor", lambda x, y: float(int(x) ^ int(y)), a, b)


def bitwise_not(a: Any) -> Any:
    """Element-wise bitwise NOT."""
    return _unary_trig("bitwise_not", "bitwise_not", lambda x: float(~int(x)), a)


def left_shift(a: Any, b: Any) -> Any:
    """Element-wise left shift."""
    return _binary_trig("left_shift", lambda x, y: float(int(x) << int(y)), a, b)


def right_shift(a: Any, b: Any) -> Any:
    """Element-wise right shift."""
    return _binary_trig("right_shift", lambda x, y: float(int(x) >> int(y)), a, b)
