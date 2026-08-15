"""
UFUNC CORE-12: Searching Operations

Provides NumPy-compatible functional API for searching:
    cp.argmax(a)
    cp.argmin(a)
    cp.where_(condition, x, y)
    cp.searchsorted(a, v)
"""

from typing import Any, Optional


def argmax(a: Any) -> int:
    """
    Return index of the maximum element.

    Args:
        a: Input array.

    Returns:
        int index of the maximum element.
    """
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)

    if a._core_array is not None:
        return a._core_array.argmax()

    # Python fallback
    from .math import _flatten

    flat = _flatten(a.to_list())
    if not flat:
        raise ValueError("argmax of empty array")
    return max(range(len(flat)), key=lambda i: flat[i])


def argmin(a: Any) -> int:
    """
    Return index of the minimum element.

    Args:
        a: Input array.

    Returns:
        int index of the minimum element.
    """
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)

    if a._core_array is not None:
        return a._core_array.argmin()

    # Python fallback
    from .math import _flatten

    flat = _flatten(a.to_list())
    if not flat:
        raise ValueError("argmin of empty array")
    return min(range(len(flat)), key=lambda i: flat[i])


def where_(condition: Any, x: Any = None, y: Any = None) -> Any:
    """
    Return elements chosen from x or y depending on condition.

    If only condition is given, equivalent to nonzero indices.

    Args:
        condition: Boolean array (0.0/1.0).
        x: Values where condition is true.
        y: Values where condition is false.

    Returns:
        ndarray with selected values, or indices if x/y not given.
    """
    from ..array import ndarray
    from .math import _flatten

    if not isinstance(condition, ndarray):
        condition = ndarray(condition)

    cond_flat = _flatten(
        condition.to_list() if hasattr(condition, "to_list") else condition
    )

    if x is None and y is None:
        # Return indices where condition is non-zero
        indices = [float(i) for i, v in enumerate(cond_flat) if v != 0.0]
        return ndarray(indices)

    if not isinstance(x, ndarray):
        x = ndarray(x if isinstance(x, (list, tuple)) else [x])
    if not isinstance(y, ndarray):
        y = ndarray(y if isinstance(y, (list, tuple)) else [y])

    x_flat = _flatten(x.to_list())
    y_flat = _flatten(y.to_list())

    # Broadcast scalars
    if len(x_flat) == 1 and len(cond_flat) > 1:
        x_flat = x_flat * len(cond_flat)
    if len(y_flat) == 1 and len(cond_flat) > 1:
        y_flat = y_flat * len(cond_flat)

    result = [xv if cv != 0.0 else yv for cv, xv, yv in zip(cond_flat, x_flat, y_flat)]
    return ndarray(result)


def searchsorted(a: Any, v: Any, side: str = "left") -> Any:
    """
    Find indices where elements should be inserted to maintain order.

    Args:
        a: Sorted input array.
        v: Values to insert.
        side: 'left' or 'right'.

    Returns:
        ndarray of insertion indices.
    """
    import bisect

    from ..array import ndarray
    from .math import _flatten

    if not isinstance(a, ndarray):
        a = ndarray(a)

    sorted_flat = _flatten(a.to_list())

    if isinstance(v, (int, float)):
        values = [v]
    else:
        if not isinstance(v, ndarray):
            v = ndarray(v)
        values = _flatten(v.to_list())

    if side == "left":
        indices = [float(bisect.bisect_left(sorted_flat, val)) for val in values]
    elif side == "right":
        indices = [float(bisect.bisect_right(sorted_flat, val)) for val in values]
    else:
        raise ValueError(f"side must be 'left' or 'right', got '{side}'")

    if len(indices) == 1:
        return int(indices[0])
    return ndarray(indices)
