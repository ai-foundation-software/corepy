"""
UFUNC CORE-50: Stacking & Splitting Operations

    cp.stack, cp.vstack, cp.hstack, cp.split, cp.array_split
    cp.tile, cp.repeat
"""

from typing import Any, List, Optional, Union


def stack(arrays: List[Any], axis: int = 0) -> Any:
    """Join arrays along a new axis."""
    from ..array import ndarray
    from .math import _flatten

    arrs = [a if isinstance(a, ndarray) else ndarray(a) for a in arrays]
    if axis != 0:
        raise NotImplementedError(f"stack along axis={axis} not yet supported")
    # Stack along axis 0: each array becomes a row
    all_flat = [_flatten(a.to_list()) for a in arrs]
    row_len = len(all_flat[0])
    for f in all_flat:
        if len(f) != row_len:
            raise ValueError("All arrays must have same length for stack")
    combined = []
    for f in all_flat:
        combined.extend(f)
    return ndarray(combined).reshape((len(arrs), row_len))


def vstack(arrays: List[Any]) -> Any:
    """Stack arrays vertically (row-wise)."""
    return stack(arrays, axis=0)


def hstack(arrays: List[Any]) -> Any:
    """Stack arrays horizontally (column-wise)."""
    from ..array import ndarray
    from .math import _flatten

    arrs = [a if isinstance(a, ndarray) else ndarray(a) for a in arrays]
    combined = []
    for a in arrs:
        combined.extend(_flatten(a.to_list()))
    return ndarray(combined)


def split(
    a: Any, indices_or_sections: Union[int, List[int]], axis: int = 0
) -> List[Any]:
    """Split array into sub-arrays."""
    from ..array import ndarray
    from .math import _flatten

    if not isinstance(a, ndarray):
        a = ndarray(a)
    flat = _flatten(a.to_list())
    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        if len(flat) % n != 0:
            raise ValueError(
                f"Cannot split array of size {len(flat)} into {n} equal parts"
            )
        chunk = len(flat) // n
        return [ndarray(flat[i * chunk : (i + 1) * chunk]) for i in range(n)]
    else:
        indices = list(indices_or_sections)
        parts = []
        prev = 0
        for idx in indices:
            parts.append(ndarray(flat[prev:idx]))
            prev = idx
        parts.append(ndarray(flat[prev:]))
        return parts


def array_split(
    a: Any, indices_or_sections: Union[int, List[int]], axis: int = 0
) -> List[Any]:
    """Split array into sub-arrays (allows unequal sizes)."""
    from ..array import ndarray
    from .math import _flatten

    if not isinstance(a, ndarray):
        a = ndarray(a)
    flat = _flatten(a.to_list())
    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        chunk = len(flat) // n
        remainder = len(flat) % n
        parts = []
        start = 0
        for i in range(n):
            size = chunk + (1 if i < remainder else 0)
            parts.append(ndarray(flat[start : start + size]))
            start += size
        return parts
    return split(a, indices_or_sections, axis)


def tile(a: Any, reps: Union[int, tuple]) -> Any:
    """Tile an array by repeating it."""
    from ..array import ndarray
    from .math import _flatten

    if not isinstance(a, ndarray):
        a = ndarray(a)
    flat = _flatten(a.to_list())
    if isinstance(reps, int):
        return ndarray(flat * reps)
    elif isinstance(reps, tuple) and len(reps) == 1:
        return ndarray(flat * reps[0])
    else:
        raise NotImplementedError("Multi-dimensional tile not yet supported")


def repeat(a: Any, repeats: int) -> Any:
    """Repeat each element of an array."""
    from ..array import ndarray
    from .math import _flatten

    if not isinstance(a, ndarray):
        a = ndarray(a)
    flat = _flatten(a.to_list())
    result = []
    for x in flat:
        result.extend([x] * repeats)
    return ndarray(result)
