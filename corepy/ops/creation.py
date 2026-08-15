"""
UFUNC CORE-50: Array Creation Functions

    cp.zeros_like, cp.ones_like, cp.full_like, cp.empty_like, cp.eye, cp.identity
"""

from typing import Any, Optional


def zeros_like(a: Any, dtype=None) -> Any:
    """Create zeros with same shape as a."""
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)
    if a._core_array is not None:
        result_ca = a._core_array.zeros_like()
        result = ndarray(result_ca.to_list(), dtype=a.dtype, backend=a.backend)
        result._core_array = result_ca
        return result
    total = 1
    for d in a.shape:
        total *= d
    return ndarray([0.0] * total, dtype=a.dtype, backend=a.backend).reshape(a.shape)


def ones_like(a: Any, dtype=None) -> Any:
    """Create ones with same shape as a."""
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)
    if a._core_array is not None:
        result_ca = a._core_array.ones_like()
        result = ndarray(result_ca.to_list(), dtype=a.dtype, backend=a.backend)
        result._core_array = result_ca
        return result
    total = 1
    for d in a.shape:
        total *= d
    return ndarray([1.0] * total, dtype=a.dtype, backend=a.backend).reshape(a.shape)


def full_like(a: Any, fill_value: float, dtype=None) -> Any:
    """Create filled array with same shape as a."""
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)
    if a._core_array is not None:
        result_ca = a._core_array.full_like(float(fill_value))
        result = ndarray(result_ca.to_list(), dtype=a.dtype, backend=a.backend)
        result._core_array = result_ca
        return result
    total = 1
    for d in a.shape:
        total *= d
    return ndarray(
        [float(fill_value)] * total, dtype=a.dtype, backend=a.backend
    ).reshape(a.shape)


def empty_like(a: Any, dtype=None) -> Any:
    """Create uninitialized array with same shape (uses zeros for safety)."""
    return zeros_like(a, dtype=dtype)


def eye(n: int, m: Optional[int] = None, dtype=None) -> Any:
    """Create n×m identity matrix."""
    from ..array import ndarray

    if m is None:
        m = n
    data = [0.0] * (n * m)
    for i in range(min(n, m)):
        data[i * m + i] = 1.0
    return ndarray(data).reshape((n, m))


def identity(n: int, dtype=None) -> Any:
    """Create n×n identity matrix."""
    return eye(n)
