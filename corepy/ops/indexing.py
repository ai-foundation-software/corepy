"""
UFUNC CORE-12: Indexing Operations

Provides NumPy-compatible functional API for advanced indexing:
    cp.take(a, indices)
    cp.boolean_index(a, mask)
"""

from typing import Any


def take(a: Any, indices: Any) -> Any:
    """
    Take elements from array at given indices.

    Args:
        a: Input array.
        indices: Array of indices to take.

    Returns:
        ndarray with selected elements.
    """
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)
    if not isinstance(indices, ndarray):
        indices = ndarray(indices if isinstance(indices, (list, tuple)) else [indices])

    if a._core_array is not None and indices._core_array is not None:
        result_ca = a._core_array.take(indices._core_array)
        result = ndarray(result_ca.to_list(), dtype=a.dtype, backend=a.backend)
        result._core_array = result_ca
        return result

    # Python fallback
    from .math import _flatten

    flat_a = _flatten(a.to_list())
    flat_idx = _flatten(indices.to_list() if hasattr(indices, "to_list") else indices)

    result = [flat_a[int(i)] for i in flat_idx]
    return ndarray(result, dtype=a.dtype, backend=a.backend)


def boolean_index(a: Any, mask: Any) -> Any:
    """
    Select elements where mask is non-zero.

    Args:
        a: Input array.
        mask: Boolean mask array (0.0/1.0).

    Returns:
        1D ndarray with selected elements.
    """
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)
    if not isinstance(mask, ndarray):
        mask = ndarray(mask if isinstance(mask, (list, tuple)) else [mask])

    if a._core_array is not None and mask._core_array is not None:
        result_ca = a._core_array.boolean_index(mask._core_array)
        result = ndarray(result_ca.to_list(), dtype=a.dtype, backend=a.backend)
        result._core_array = result_ca
        return result

    # Python fallback
    from .math import _flatten

    flat_a = _flatten(a.to_list())
    flat_mask = _flatten(mask.to_list() if hasattr(mask, "to_list") else mask)

    result = [v for v, m in zip(flat_a, flat_mask) if m != 0.0]
    return ndarray(result, dtype=a.dtype, backend=a.backend)
