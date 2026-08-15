"""
Test suite for array concatenation.
"""

import numpy as np
import pytest

import corepy as cp


def test_concatenate_1d():
    """Test concatenating two 1D arrays."""
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5])
    res = cp.concatenate((a, b))

    assert isinstance(res, cp.ndarray)
    assert res.shape == (5,)
    np.testing.assert_array_equal(
        res.to_list(), np.array([1, 2, 3, 4, 5], dtype=np.float32)
    )


def test_concatenate_list_input():
    """Test concatenating lists directly."""
    # NumPy supports np.concatenate((list1, list2))
    # CorePy implementation needs to handle this or expect arrays.
    # Plan says: "Convert all inputs to ndarray if needed".
    res = cp.concatenate(([1, 2], [3]))
    assert res.shape == (3,)
    assert res.to_list() == [1.0, 2.0, 3.0]


def test_concatenate_2d_axis0():
    """Test concatenating 2D arrays along axis 0 (default)."""
    a = cp.array([[1, 2], [3, 4]])
    b = cp.array([[5, 6]])
    res = cp.concatenate((a, b), axis=0)

    assert res.shape == (3, 2)
    expected = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    np.testing.assert_array_equal(res.to_list(), expected)


def test_concatenate_2d_axis1():
    """Test concatenating 2D arrays along axis 1."""
    a = cp.array([[1, 2], [3, 4]])
    b = cp.array([[5], [6]])
    res = cp.concatenate((a, b), axis=1)

    assert res.shape == (2, 3)
    expected = np.array([[1, 2, 5], [3, 4, 6]], dtype=np.float32)
    np.testing.assert_array_equal(res.to_list(), expected)
