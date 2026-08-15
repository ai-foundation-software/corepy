from typing import Any, Optional, Tuple, Union

from .array import ndarray
from .backend.types import DataType

try:
    from . import _corepy_rust
except ImportError:
    _corepy_rust = None


def rand(*shape: Any, seed: int = 42, algo: str = "pcg64") -> ndarray:
    """
    Generate an array of uniform random float32 values in [0.0, 1.0).

    Args:
        *shape: Dimensions of the array (passed as integers or a single tuple)
        seed: Random seed for reproducibility
        algo: 'pcg64' or 'xoshiro'
    """
    if _corepy_rust is None:
        raise ImportError("Rust backend is required for random generation")

    # Handle both rand(5, 5) and rand((5, 5))
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        actual_shape = tuple(shape[0])
    else:
        actual_shape = tuple(shape)

    if not actual_shape:
        actual_shape = (1,)

    rng_algo = _corepy_rust.RngAlgorithm.PCG64
    if algo.lower() == "xoshiro":
        rng_algo = _corepy_rust.RngAlgorithm.Xoshiro256PP

    ct = _corepy_rust.random_uniform_f32(list(actual_shape), seed, rng_algo)
    return ndarray._wrap_core_array(ct, dtype=DataType.FLOAT32)


def randn(*shape: Any, seed: int = 42, algo: str = "pcg64") -> ndarray:
    """
    Generate an array of standard normal random float32 values.

    Args:
        *shape: Dimensions of the array (passed as integers or a single tuple)
        seed: Random seed for reproducibility
        algo: 'pcg64' or 'xoshiro'
    """
    if _corepy_rust is None:
        raise ImportError("Rust backend is required for random generation")

    # Handle both randn(5, 5) and randn((5, 5))
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        actual_shape = tuple(shape[0])
    else:
        actual_shape = tuple(shape)

    if not actual_shape:
        actual_shape = (1,)

    rng_algo = _corepy_rust.RngAlgorithm.PCG64
    if algo.lower() == "xoshiro":
        rng_algo = _corepy_rust.RngAlgorithm.Xoshiro256PP

    ct = _corepy_rust.random_normal_f32(list(actual_shape), seed, rng_algo)
    return ndarray._wrap_core_array(ct, dtype=DataType.FLOAT32)


def randint(
    low: int,
    high: Optional[int] = None,
    shape: Union[int, Tuple[int, ...]] = 1,
    seed: int = 42,
    algo: str = "pcg64",
) -> ndarray:
    """Return random integers from low (inclusive) to high (exclusive)."""
    if high is None:
        high = low
        low = 0
    import random

    if isinstance(shape, int):
        shape = (shape,)

    count = 1
    for d in shape:
        count *= d

    r = random.Random(seed)
    data = [r.randint(low, high - 1) for _ in range(count)]

    return ndarray(data, dtype=DataType.INT32).reshape(shape)
