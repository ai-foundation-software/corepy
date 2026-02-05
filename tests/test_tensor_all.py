"""
Test suite for tensor.all() reduction operation.

Tests the complete 3-layer execution: Python → Rust → C++
"""

from corepy.backend.types import DataType
from corepy.tensor import Tensor


def test_all_true():
    """All elements are True."""
    t = Tensor([True, True, True], dtype=DataType.BOOL)
    result = t.all()
    assert result._backing_data == [True]


def test_all_false():
    """All elements are False."""
    t = Tensor([False, False, False], dtype=DataType.BOOL)
    result = t.all()
    assert result._backing_data == [False]


def test_mixed_values():
    """Mixed True/False returns False."""
    t = Tensor([True, False, True], dtype=DataType.BOOL)
    result = t.all()
    assert result._backing_data == [False]


def test_single_false():
    """Single False among many True values."""
    t = Tensor([True] * 100 + [False], dtype=DataType.BOOL)
    result = t.all()
    assert result._backing_data == [False]


def test_single_true():
    """Single True value."""
    t = Tensor([True], dtype=DataType.BOOL)
    result = t.all()
    assert result._backing_data == [True]


def test_empty_tensor():
    """Empty tensor returns True (vacuous truth)."""
    t = Tensor([], dtype=DataType.BOOL)
    result = t.all()
    assert result._backing_data == [True]


def test_large_tensor_true():
    """Large tensor (triggers SIMD path) - all True."""
    # 1000 elements will trigger AVX2 optimization (processes 32 at a time)
    t = Tensor([True] * 1000, dtype=DataType.BOOL)
    result = t.all()
    assert result._backing_data == [True]


def test_large_tensor_false():
    """Large tensor (triggers SIMD path) - contains False."""
    t = Tensor([True] * 500 + [False] + [True] * 499, dtype=DataType.BOOL)
    result = t.all()
    assert result._backing_data == [False]


def test_truthy_integers():
    """Test with truthy integer values (non-zero)."""
    # In Python, non-zero integers are truthy
    t = Tensor([1, 2, 3, 4, 5])
    result = t.all()
    # All non-zero integers should be truthy
    assert result._backing_data == [True]


def test_zero_in_integers():
    """Test with zero (falsy) in integer list."""
    t = Tensor([1, 2, 0, 4, 5])
    result = t.all()
    # Contains zero (falsy), should return False
    assert result._backing_data == [False]


def test_boundary_31_elements():
    """Test with 31 elements (just under AVX2 chunk size of 32)."""
    t = Tensor([True] * 31, dtype=DataType.BOOL)
    result = t.all()
    assert result._backing_data == [True]


def test_boundary_32_elements():
    """Test with exactly 32 elements (one AVX2 chunk)."""
    t = Tensor([True] * 32, dtype=DataType.BOOL)
    result = t.all()
    assert result._backing_data == [True]


def test_boundary_33_elements():
    """Test with 33 elements (one AVX2 chunk + 1 remainder)."""
    t = Tensor([True] * 33, dtype=DataType.BOOL)
    result = t.all()
    assert result._backing_data == [True]
