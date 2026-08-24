"""
UFUNC CORE-50: Trigonometric & Hyperbolic Operations

Provides NumPy-compatible API:
    cp.sin, cp.cos, cp.tan, cp.arcsin, cp.arccos, cp.arctan, cp.arctan2
    cp.sinh, cp.cosh, cp.tanh, cp.arcsinh, cp.arccosh, cp.arctanh
    cp.degrees, cp.radians, cp.hypot
"""

import math
from typing import Any

from .domain_guards import safe_eval_binary, safe_eval_unary
from .ufunc_engine import _ensure_array, ufunc_binary, ufunc_unary


def _unary_trig(op_name: str, core_method: str, py_fn, a: Any) -> Any:
    """Generic unary trig helper with robust domain error handling."""
    from ..array import ndarray
    from .domain_guards import safe_eval_unary
    from .math import _flatten

    a = _ensure_array(a)
    flat = _flatten(a.to_list() if hasattr(a, "to_list") else a)
    for x in flat:
        safe_eval_unary(op_name, py_fn, float(x))

    if a._core_array is not None:
        fn = getattr(a._core_array, core_method, None)
        if fn is not None:
            try:
                result_ca = fn()
                return ndarray._from_core_array(
                    result_ca, dtype=a.dtype, backend=a.backend
                )
            except Exception:
                pass

    res = [safe_eval_unary(op_name, py_fn, float(x)) for x in flat]
    return ndarray(res, dtype=a.dtype, backend=a.backend)


def _binary_trig(core_method: str, py_fn, a: Any, b: Any) -> Any:
    """Generic binary trig helper with robust domain error handling."""
    from ..array import ndarray
    from .ufunc_engine import _broadcast_pair

    a, b = _broadcast_pair(a, b)
    if a._core_array is not None and b._core_array is not None:
        fn = getattr(a._core_array, core_method, None)
        if fn is not None:
            try:
                result_ca = fn(b._core_array)
                return ndarray._from_core_array(
                    result_ca, dtype=a.dtype, backend=a.backend
                )
            except Exception:
                pass
    from .math import _flatten

    flat_a = _flatten(a.to_list() if hasattr(a, "to_list") else a)
    flat_b = _flatten(b.to_list() if hasattr(b, "to_list") else b)
    res = [
        safe_eval_binary(core_method, py_fn, float(x), float(y))
        for x, y in zip(flat_a, flat_b)
    ]
    return ndarray(res, dtype=a.dtype, backend=a.backend)


# Trigonometric
def sin(a: Any) -> Any:
    """Element-wise sine."""
    return _unary_trig("sin", "sin", math.sin, a)


def cos(a: Any) -> Any:
    """Element-wise cosine."""
    return _unary_trig("cos", "cos", math.cos, a)


def tan(a: Any) -> Any:
    """Element-wise tangent."""
    return _unary_trig("tan", "tan", math.tan, a)


def arcsin(a: Any) -> Any:
    """Element-wise inverse sine."""
    return _unary_trig("arcsin", "arcsin", math.asin, a)


def arccos(a: Any) -> Any:
    """Element-wise inverse cosine."""
    return _unary_trig("arccos", "arccos", math.acos, a)


def arctan(a: Any) -> Any:
    """Element-wise inverse tangent."""
    return _unary_trig("arctan", "arctan", math.atan, a)


def arctan2(a: Any, b: Any) -> Any:
    """Element-wise arctan2(y, x)."""
    return _binary_trig("arctan2", math.atan2, a, b)


# Hyperbolic
def sinh(a: Any) -> Any:
    """Element-wise hyperbolic sine."""
    return _unary_trig("sinh", "sinh", math.sinh, a)


def cosh(a: Any) -> Any:
    """Element-wise hyperbolic cosine."""
    return _unary_trig("cosh", "cosh", math.cosh, a)


def tanh(a: Any) -> Any:
    """Element-wise hyperbolic tangent."""
    return _unary_trig("tanh", "tanh", math.tanh, a)


def arcsinh(a: Any) -> Any:
    """Element-wise inverse hyperbolic sine."""
    return _unary_trig("arcsinh", "arcsinh", math.asinh, a)


def arccosh(a: Any) -> Any:
    """Element-wise inverse hyperbolic cosine."""
    return _unary_trig("arccosh", "arccosh", math.acosh, a)


def arctanh(a: Any) -> Any:
    """Element-wise inverse hyperbolic tangent."""
    return _unary_trig("arctanh", "arctanh", math.atanh, a)


# Angle conversion
def degrees(a: Any) -> Any:
    """Convert radians to degrees."""
    return _unary_trig("degrees", "degrees_op", math.degrees, a)


def radians(a: Any) -> Any:
    """Convert degrees to radians."""
    return _unary_trig("radians", "radians_op", math.radians, a)


def hypot(a: Any, b: Any) -> Any:
    """Element-wise hypotenuse: sqrt(a² + b²)."""
    return _binary_trig("hypot", math.hypot, a, b)
