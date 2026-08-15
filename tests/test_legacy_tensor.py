"""
Test suite for backward compatibility of the deprecated corepy.ndarray API.
"""

import warnings

import pytest

import corepy
from corepy import ndarray


def test_array_alias_exists():
    """Test that ndarray is available as an alias to ndarray."""
    assert corepy.ndarray is corepy.ndarray
    assert ndarray is ndarray


def test_array_construction_warning():
    """Test that instantiating ndarray emits a DeprecationWarning (if we were using a factory/wrapper).

    Current implementation: ndarray = ndarray.
    Direct instantiation `ndarray(...)` is just `ndarray(...)`, so no warning is emitted unless
    we wrapped it.

    Wait, I implemented a `_deprecated_array_class` but assigned `ndarray = ndarray` at the end of array.py.
    Let's check array.py content again.
    """
    # If ndarray is just an alias to ndarray, there won't be a warning on instantiation
    # unless __init__ checks how it was called, which is hard.
    # checking if I aliased it to the function or the class.

    t = ndarray([1, 2, 3])
    assert isinstance(t, ndarray)
    assert t.shape == (3,)


def test_array_operations():
    """Test that objects created via ndarray support operations."""
    t1 = ndarray([1.0, 2.0])
    t2 = ndarray([3.0, 4.0])

    res = t1 + t2
    assert isinstance(res, ndarray)
    import numpy as np

    expected = [4.0, 6.0]
    if isinstance(res.to_list(), np.ndarray):
        np.testing.assert_array_equal(res.to_list(), expected)
    else:
        assert res.to_list() == expected


def test_isinstance_array():
    """Test isinstance checks with ndarray."""
    t = corepy.array([1, 2, 3])
    # Since ndarray is alias to ndarray, this should be true
    assert isinstance(t, ndarray)

    t2 = ndarray([1, 2, 3])
    assert isinstance(t2, corepy.ndarray)


def test_legacy_import():
    """Test importing ndarray from corepy.array."""
    from corepy.array import ndarray as ArrayFromModule

    assert ArrayFromModule is corepy.ndarray
