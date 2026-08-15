"""
UFUNC LINALG-5: Linear Algebra

dot, matmul, linalg.inv, linalg.det, linalg.norm
"""

import math
from typing import Any


def _ensure_2d(a: Any) -> Any:
    from .array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)
    if len(a.shape) != 2:
        raise ValueError("Matrix must be 2D")
    return a


def inv(a: Any) -> Any:
    """Compute the (multiplicative) inverse of a matrix."""
    a = _ensure_2d(a)
    if a._core_array is not None:
        return a._wrap_core_array(a._core_array.linalg_inv())

    try:
        from ._corepy_rust import _RustCoreArray as _CT
        from .ops.math import _flatten

        flat = list(_flatten(a.to_list()))
        ct = _CT(flat, list(a.shape))
        inv_ct = ct.linalg_inv()
        return a._wrap_core_array(inv_ct)
    except Exception as e:
        raise NotImplementedError(
            f"Inverse of matrix shape {a.shape} failed: {e}"
        ) from e


def det(a: Any) -> float:
    """Compute the determinant of an array."""
    a = _ensure_2d(a)
    if a._core_array is not None:
        return float(a._core_array.linalg_det())

    try:
        from ._corepy_rust import _RustCoreArray as _CT
        from .ops.math import _flatten

        flat = list(_flatten(a.to_list()))
        ct = _CT(flat, list(a.shape))
        return float(ct.linalg_det())
    except Exception as e:
        raise NotImplementedError(
            f"Determinant of matrix shape {a.shape} failed: {e}"
        ) from e


def norm(a: Any) -> float:
    """Matrix or vector norm (Frobenius / L2)."""
    from .array import ndarray
    from .ops.math import _flatten

    if not isinstance(a, ndarray):
        a = ndarray(a)

    if a._core_array is not None:
        return float(a._core_array.linalg_norm())

    try:
        from ._corepy_rust import _RustCoreArray as _CT

        flat = list(_flatten(a.to_list()))
        ct = _CT(flat, list(a.shape))
        return float(ct.linalg_norm())
    except Exception:
        flat = list(_flatten(a.to_list()))
        return math.sqrt(sum(float(x) * float(x) for x in flat))
