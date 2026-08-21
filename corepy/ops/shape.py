"""
UFUNC CORE-50: Shape Manipulation Operations
"""

from typing import Any


def reshape(a: Any, newshape: tuple) -> Any:
    """Give a new shape to an array without changing its data."""
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)
    return a.reshape(newshape)


def transpose(a: Any, *axes) -> Any:
    """Reverse or permute the axes of an array."""
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)
    return a.transpose(*axes)


def ravel(a: Any) -> Any:
    """Return a contiguous flattened array."""
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)
    if hasattr(a, "ravel"):
        return a.ravel()
    return a.reshape((a.size,))


def flatten(a: Any) -> Any:
    """Return a copy of the array collapsed into one dimension."""
    return ravel(a)


def squeeze(a: Any, axis: Any = None) -> Any:
    """Remove single-dimensional entries from the shape of an array."""
    from ..array import ndarray

    if not isinstance(a, ndarray):
        a = ndarray(a)
    if axis is None:
        new_shape = tuple(s for s in a.shape if s != 1)
        if not new_shape:
            new_shape = (1,)
    elif isinstance(axis, int):
        axes = (axis % len(a.shape),)
        new_shape = tuple(s for i, s in enumerate(a.shape) if i not in axes or s != 1)
    else:
        axes = tuple(ax % len(a.shape) for ax in axis)
        new_shape = tuple(s for i, s in enumerate(a.shape) if i not in axes or s != 1)
    return a.reshape(new_shape)
