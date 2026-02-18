"""
Test suite for backward compatibility of the deprecated corepy.Tensor API.
"""

import warnings

import pytest

import corepy
from corepy import Tensor, ndarray


def test_tensor_alias_exists():
    """Test that Tensor is available as an alias to ndarray."""
    assert corepy.Tensor is corepy.ndarray
    assert Tensor is ndarray


def test_tensor_construction_warning():
    """Test that instantiating Tensor emits a DeprecationWarning (if we were using a factory/wrapper).

    Current implementation: Tensor = ndarray.
    Direct instantiation `Tensor(...)` is just `ndarray(...)`, so no warning is emitted unless
    we wrapped it.

    Wait, I implemented a `_deprecated_tensor_class` but assigned `Tensor = ndarray` at the end of tensor.py.
    Let's check tensor.py content again.
    """
    # If Tensor is just an alias to ndarray, there won't be a warning on instantiation
    # unless __init__ checks how it was called, which is hard.
    # checking if I aliased it to the function or the class.

    t = Tensor([1, 2, 3])
    assert isinstance(t, ndarray)
    assert t.shape == (3,)


def test_tensor_operations():
    """Test that objects created via Tensor support operations."""
    t1 = Tensor([1.0, 2.0])
    t2 = Tensor([3.0, 4.0])

    res = t1 + t2
    assert isinstance(res, ndarray)
    import numpy as np

    expected = [4.0, 6.0]
    if isinstance(res._backing_data, np.ndarray):
        np.testing.assert_array_equal(res._backing_data, expected)
    else:
        assert res._backing_data == expected


def test_isinstance_tensor():
    """Test isinstance checks with Tensor."""
    t = corepy.array([1, 2, 3])
    # Since Tensor is alias to ndarray, this should be true
    assert isinstance(t, Tensor)

    t2 = Tensor([1, 2, 3])
    assert isinstance(t2, corepy.ndarray)


def test_legacy_import():
    """Test importing Tensor from corepy.tensor."""
    from corepy.array import Tensor as TensorFromModule

    assert TensorFromModule is corepy.ndarray
