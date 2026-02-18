"""
Test suite for Python fallback kernels in corepy/ops/math.py.
These kernels are used when the Rust/C++ backend is unavailable.
Targeting them directly improves code coverage and ensures fallback reliability.
"""

import math

import pytest

from corepy.ops.math import (
    _flatten,
    _reshape,
    cpu_add,
    cpu_all,
    cpu_any,
    cpu_div,
    cpu_matmul,
    cpu_max,
    cpu_mean,
    cpu_min,
    cpu_mul,
    cpu_std,
    cpu_sub,
    cpu_sum,
)


def test_flatten_types():
    """Test flattening of various input types."""
    # List
    assert _flatten([[1, 2], [3]]) == [1.0, 2.0, 3.0]
    # Tuple
    assert _flatten((1, (2, 3))) == [1.0, 2.0, 3.0]
    # Scalar
    assert _flatten(10) == [10.0]


def test_reshape():
    """Test reshaping flat list."""
    flat = [1.0, 2.0, 3.0, 4.0]
    # 2x2
    assert _reshape(flat, (2, 2)) == [[1.0, 2.0], [3.0, 4.0]]
    # 1D
    assert _reshape(flat, (4,)) == [1.0, 2.0, 3.0, 4.0]
    # Scalar
    assert _reshape([42.0], ()) == 42.0


def test_cpu_arithmetic():
    """Test basic arithmetic kernels."""
    a = [10, 20]
    b = [1, 2]
    assert cpu_add(a, b) == [11.0, 22.0]
    assert cpu_sub(a, b) == [9.0, 18.0]
    assert cpu_mul(a, b) == [10.0, 40.0]
    assert cpu_div(a, b) == [10.0, 10.0]


def test_cpu_arithmetic_mismatch():
    """Test shape mismatch handling."""
    with pytest.raises(ValueError):
        cpu_add([1], [1, 2])


def test_cpu_matmul_1d():
    """Test dot product."""
    a = [1, 2, 3]
    b = [4, 5, 6]
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert cpu_matmul(a, b) == 32.0


def test_cpu_matmul_2d():
    """Test matrix multiplication."""
    # 2x2 @ 2x2
    # [[1, 2],  @  [[1, 0],
    #  [3, 4]]      [0, 1]]
    a = [[1, 2], [3, 4]]
    b = [[1, 0], [0, 1]]
    shape = (2, 2)
    res = cpu_matmul(a, b, shape_a=shape, shape_b=shape)
    assert res == [[1.0, 2.0], [3.0, 4.0]]


def test_cpu_matmul_mismatch():
    """Test matmul dimension mismatch."""
    # 1x2 @ 3x1 -> error
    a = [[1, 2]]
    b = [[1], [2], [3]]
    with pytest.raises(ValueError):
        cpu_matmul(a, b, shape_a=(1, 2), shape_b=(3, 1))


def test_cpu_reductions():
    """Test reduction kernels."""
    data = [1, 2, 3, 4, 5]
    assert cpu_sum(data) == 15.0
    assert cpu_mean(data) == 3.0
    assert cpu_max(data) == 5.0
    assert cpu_min(data) == 1.0

    # Std: mean=3. var=((1-3)^2 + ...)/5 = (4+1+0+1+4)/5 = 2. sqrt(2)=1.414...
    assert abs(cpu_std(data) - math.sqrt(2)) < 1e-6


def test_cpu_boolean():
    """Test boolean reductions."""
    assert cpu_all([True, True]) == True
    assert cpu_all([True, False]) == False
    assert cpu_any([False, False]) == False
    assert cpu_any([True, False]) == True
