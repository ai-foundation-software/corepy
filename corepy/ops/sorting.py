"""
UFUNC CORE-12: Sorting Operations

Provides NumPy-compatible functional API for sorting:
    cp.sort(a)
    cp.argsort(a)
    cp.stable_sort(a)
"""

from typing import Any


def sort(a: Any, descending: bool = False) -> Any:
    """
    Return a sorted copy of the array.

    Args:
        a: Input array.
        descending: If True, sort in descending order.

    Returns:
        Sorted ndarray.
    """
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)

    if a._core_array is not None:
        result_ca = a._core_array.sort(descending)
        result = ndarray(result_ca.to_list(), dtype=a.dtype, backend=a.backend)
        result._core_array = result_ca
        return result

    # Python fallback
    from .math import _flatten

    flat = _flatten(a.to_list())
    flat.sort(reverse=descending)
    return ndarray(flat, dtype=a.dtype, backend=a.backend)


def argsort(a: Any, descending: bool = False) -> Any:
    """
    Return indices that would sort the array.

    Args:
        a: Input array.
        descending: If True, sort in descending order.

    Returns:
        ndarray of indices.
    """
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)

    if a._core_array is not None:
        result_ca = a._core_array.argsort(descending)
        result = ndarray(result_ca.to_list(), dtype=a.dtype, backend=a.backend)
        result._core_array = result_ca
        return result

    # Python fallback
    from .math import _flatten

    flat = _flatten(a.to_list())
    indices = sorted(range(len(flat)), key=lambda i: flat[i], reverse=descending)
    return ndarray([float(i) for i in indices], dtype=a.dtype, backend=a.backend)


def stable_sort(a: Any, descending: bool = False) -> Any:
    """
    Return a stably sorted copy of the array.

    Uses Python's timsort (stable) as fallback; Rust's sort_by is also stable.
    """
    return sort(a, descending=descending)
