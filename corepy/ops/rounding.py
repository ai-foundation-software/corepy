"""
UFUNC CORE-50: Rounding, Sign, and Clip Operations

    cp.floor, cp.ceil, cp.round_, cp.trunc, cp.rint
    cp.sign, cp.clip, cp.clamp, cp.copysign
"""

import math
from typing import Any

from .trigonometry import _binary_trig, _unary_trig


def floor(a: Any) -> Any:
    """Element-wise floor."""
    return _unary_trig("floor", "floor_op", math.floor, a)


def ceil(a: Any) -> Any:
    """Element-wise ceiling."""
    return _unary_trig("ceil", "ceil_op", math.ceil, a)


def round_(a: Any) -> Any:
    """Element-wise rounding to nearest integer."""
    return _unary_trig("round", "round_op", round, a)


def trunc(a: Any) -> Any:
    """Element-wise truncation toward zero."""
    return _unary_trig("trunc", "trunc_op", math.trunc, a)


def rint(a: Any) -> Any:
    """Round to nearest even integer."""
    return _unary_trig("rint", "rint", lambda x: round(x), a)


def sign(a: Any) -> Any:
    """Element-wise sign: -1, 0, or 1."""
    return _unary_trig(
        "sign", "sign_op", lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0), a
    )


def clip(a: Any, a_min: float, a_max: float) -> Any:
    """Clip values to [a_min, a_max]."""
    from ..array import ndarray
    from .ufunc_engine import _ensure_array

    a = _ensure_array(a)
    if a._core_array is not None:
        result_ca = a._core_array.clip(float(a_min), float(a_max))
        result = ndarray(result_ca.to_list(), dtype=a.dtype, backend=a.backend)
        result._core_array = result_ca
        return result
    from .math import _flatten

    flat = _flatten(a.to_list() if hasattr(a, "to_list") else a)
    return ndarray(
        [max(a_min, min(a_max, x)) for x in flat], dtype=a.dtype, backend=a.backend
    )


def clamp(a: Any, a_min: float, a_max: float) -> Any:
    """Alias for clip."""
    return clip(a, a_min, a_max)


def copysign(a: Any, b: Any) -> Any:
    """Element-wise copysign: magnitude of a, sign of b."""
    return _binary_trig("copysign", math.copysign, a, b)
