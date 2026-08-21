"""
UFUNC CORE-50: Reduction Operations

    cp.prod, cp.std, cp.var, cp.cumsum, cp.cumprod
    cp.nansum, cp.nanmean, cp.nanmax, cp.nanmin
"""

import builtins
import math
from typing import Any


def _ensure(a):
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a if isinstance(a, (list, tuple)) else [a])
    return a


def prod(a: Any) -> float:
    """Product of all elements."""
    a = _ensure(a)
    if a._core_array is not None:
        return a._core_array.prod()
    from .math import _flatten

    flat = _flatten(a.to_list())
    result = 1.0
    for x in flat:
        result *= x
    return result


def std(a: Any, axis: Any = None, ddof: int = 0, keepdims: bool = False) -> float:
    """Standard deviation of elements."""
    a = _ensure(a)
    if ddof == 0 and axis is None and a._core_array is not None:
        try:
            return a._core_array.std_dev()
        except Exception:
            pass
    from .math import _flatten

    flat = _flatten(a.to_list())
    n = len(flat)
    if n == 0:
        return float("nan")
    mean = builtins.sum(flat) / n
    denom = builtins.max(1, n - ddof) if n > ddof else n
    v = builtins.sum((x - mean) ** 2 for x in flat) / denom
    return math.sqrt(v)


def var(a: Any, axis: Any = None, ddof: int = 0, keepdims: bool = False) -> float:
    """Variance of elements."""
    a = _ensure(a)
    if ddof == 0 and axis is None and a._core_array is not None:
        try:
            return a._core_array.var()
        except Exception:
            pass
    from .math import _flatten

    flat = _flatten(a.to_list())
    n = len(flat)
    if n == 0:
        return float("nan")
    mean = builtins.sum(flat) / n
    denom = builtins.max(1, n - ddof) if n > ddof else n
    return builtins.sum((x - mean) ** 2 for x in flat) / denom


def cumsum(a: Any) -> Any:
    """Cumulative sum."""
    a = _ensure(a)
    from ..array import ndarray
    from .math import _flatten

    flat = _flatten(a.to_list())
    out = []
    curr = 0.0
    for x in flat:
        curr += x
        out.append(curr)
    return ndarray(out, dtype=a.dtype, backend=a.backend)


def cumprod(a: Any) -> Any:
    """Cumulative product."""
    a = _ensure(a)
    from ..array import ndarray
    from .math import _flatten

    flat = _flatten(a.to_list())
    out = []
    curr = 1.0
    for x in flat:
        curr *= x
        out.append(curr)
    return ndarray(out, dtype=a.dtype, backend=a.backend)


def nansum(a: Any) -> float:
    """Sum ignoring NaN."""
    a = _ensure(a)
    if a._core_array is not None:
        return a._core_array.nansum()
    from .math import _flatten

    flat = _flatten(a.to_list())
    return builtins.sum(x for x in flat if not math.isnan(x))


def nanmean(a: Any) -> float:
    """Mean ignoring NaN."""
    a = _ensure(a)
    if a._core_array is not None:
        return a._core_array.nanmean()
    from .math import _flatten

    flat = _flatten(a.to_list())
    valid = [x for x in flat if not math.isnan(x)]
    if not valid:
        return float("nan")
    return builtins.sum(valid) / len(valid)


def nanmax(a: Any) -> float:
    """Max ignoring NaN."""
    a = _ensure(a)
    if a._core_array is not None:
        return a._core_array.nanmax()
    from .math import _flatten

    flat = _flatten(a.to_list())
    valid = [x for x in flat if not math.isnan(x)]
    return builtins.max(valid) if valid else float("-inf")


def nanmin(a: Any) -> float:
    """Min ignoring NaN."""
    a = _ensure(a)
    if a._core_array is not None:
        return a._core_array.nanmin()
    from .math import _flatten

    flat = _flatten(a.to_list())
    valid = [x for x in flat if not math.isnan(x)]
    return builtins.min(valid) if valid else float("inf")


def sum(a: Any) -> float:
    """Sum array elements over a given axis."""
    a = _ensure(a)
    if hasattr(a, "sum"):
        return a.sum()
    from .math import _flatten

    return math.fsum(_flatten(a.to_list()))


def mean(a: Any) -> float:
    """Compute the arithmetic mean along the specified axis."""
    a = _ensure(a)
    if hasattr(a, "mean"):
        return a.mean()
    from .math import _flatten

    data = _flatten(a.to_list())
    if not data:
        return 0.0
    return math.fsum(data) / len(data)


def max(a: Any) -> float:
    """Return the maximum of an array or maximum along an axis."""
    a = _ensure(a)
    if hasattr(a, "max"):
        return a.max()
    from .math import _flatten

    return builtins.max(_flatten(a.to_list()))


def min(a: Any) -> float:
    """Return the minimum of an array or minimum along an axis."""
    a = _ensure(a)
    if hasattr(a, "min"):
        return a.min()
    from .math import _flatten

    return builtins.min(_flatten(a.to_list()))


import builtins
