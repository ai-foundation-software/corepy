from typing import List, Tuple


def broadcast_shapes(
    shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Computes the broadcasted shape of two arrays.
    """
    ndim_a = len(shape_a)
    ndim_b = len(shape_b)
    ndim = max(ndim_a, ndim_b)

    # Right-align shapes
    aligned_a = (1,) * (ndim - ndim_a) + shape_a
    aligned_b = (1,) * (ndim - ndim_b) + shape_b

    result_shape = []
    for da, db in zip(aligned_a, aligned_b):
        if da == db:
            result_shape.append(da)
        elif da == 1:
            result_shape.append(db)
        elif db == 1:
            result_shape.append(da)
        else:
            raise ValueError(f"Shape mismatch: {shape_a} vs {shape_b}")

    return tuple(result_shape)


def get_c_strides(shape: Tuple[int, ...]) -> List[int]:
    """Compute standard C-contiguous strides for a shape."""
    if not shape:
        return []
    strides = [0] * len(shape)
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return strides


def compute_broadcast_strides(
    original_shape: Tuple[int, ...],
    target_shape: Tuple[int, ...],
    original_strides: List[int],
) -> List[int]:
    """
    Computes strides for broadcasted shape.
    If a dimension is broadcasted (size 1 -> N), stride is 0.
    """
    # Right-align
    ndim_orig = len(original_shape)
    ndim_target = len(target_shape)
    offset = ndim_target - ndim_orig

    result_strides = [0] * ndim_target

    for i in range(ndim_target):
        # Index in original shape
        orig_idx = i - offset

        if orig_idx < 0:
            # Dimension added by expansion (e.g. (3) -> (1,3)), effectively size 1 broadcasted to target size
            # Stride is 0
            result_strides[i] = 0
        else:
            dim_size = original_shape[orig_idx]
            target_size = target_shape[i]

            if dim_size == target_size:
                # Normal dimension
                result_strides[i] = original_strides[orig_idx]
            elif dim_size == 1:
                # Broadcasted dimension
                result_strides[i] = 0
            else:
                raise ValueError("Should not happen if broadcast_shapes succeeded")

    return result_strides
