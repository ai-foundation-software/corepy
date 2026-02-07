"""
Test suite for array slicing and indexing.
"""

import numpy as np
import pytest

import corepy as cp


def test_getitem_integer():
    """Test accessing single element via integer index."""
    arr = cp.array([10, 20, 30])
    val = arr[1]
    # Should return a scalar-like object or scalar
    # Current implementation of _get_scalar_value returns float
    # But __getitem__ typically returns an element.
    # If we follow NumPy, arr[0] returns a scalar type (e.g. np.float64).
    # For now, let's look at what our plan says: "Return new ndarray ... or scalar"
    # Let's assert equality with value.
    assert val == 20


def test_getitem_slice_basic():
    """Test basic slicing arr[start:stop]."""
    arr = cp.array([10, 20, 30, 40, 50])
    sub = arr[1:4]

    assert isinstance(sub, cp.ndarray)
    assert sub.shape == (3,)
    np.testing.assert_array_equal(
        sub.to_numpy(), np.array([20, 30, 40], dtype=np.float32)
    )


def test_getitem_slice_step():
    """Test slicing with step arr[::step]."""
    arr = cp.array([1, 2, 3, 4, 5])
    sub = arr[::2]

    assert isinstance(sub, cp.ndarray)
    assert sub.shape == (3,)
    np.testing.assert_array_equal(sub.to_numpy(), np.array([1, 3, 5], dtype=np.float32))


def test_getitem_2d_row():
    """Test accessing a specific row in 2D array."""
    arr = cp.array([[1, 2], [3, 4]])
    row = arr[1]

    assert isinstance(row, cp.ndarray)
    assert row.shape == (2,)
    np.testing.assert_array_equal(row.to_numpy(), np.array([3, 4], dtype=np.float32))


def test_getitem_2d_element():
    """Test accessing single element in 2D array."""
    arr = cp.array([[1, 2], [3, 4]])
    # arr[1][0]
    val = arr[1][0]
    assert val == 3


def test_getitem_tuple_indexing():
    """Test arr[row, col] format."""
    arr = cp.array([[1, 2], [3, 4]])
    val = arr[1, 0]
    assert val == 3
