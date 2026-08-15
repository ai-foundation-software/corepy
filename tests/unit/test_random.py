import numpy as np
import pytest

import corepy as cp


def test_rand_shape_and_type():
    """Test rand() returns correct shape and datatype."""
    arr = cp.random.rand((3, 4))
    assert arr.shape == (3, 4)
    assert arr.dtype == cp.float32

    # Integer shape tests
    arr2 = cp.random.rand(5)
    assert arr2.shape == (5,)


def test_rand_bounds():
    """Test rand() returns values between 0.0 and 1.0."""
    arr = cp.random.rand(100)
    data = np.array(arr.to_list())
    assert np.all(data >= 0.0)
    assert np.all(data < 1.0)


def test_rand_reproducibility():
    """Test that rand() with the same seed produces identical arrays."""
    arr1 = cp.random.rand((5, 5), seed=42)
    arr2 = cp.random.rand((5, 5), seed=42)
    arr3 = cp.random.rand((5, 5), seed=99)

    np.testing.assert_array_equal(arr1.to_list(), arr2.to_list())

    # Very unlikely to be exactly equal
    assert not np.array_equal(arr1.to_list(), arr3.to_list())


def test_rand_algo_xoshiro():
    """Test rand() works with the xoshiro algorithm."""
    arr = cp.random.rand((10,), seed=123, algo="xoshiro")
    assert arr.shape == (10,)
    assert arr.dtype == cp.float32


def test_randn_shape_and_type():
    """Test randn() returns correct shape and datatype."""
    arr = cp.random.randn((3, 4))
    assert arr.shape == (3, 4)
    assert arr.dtype == cp.float32


def test_randn_reproducibility():
    """Test that randn() with the same seed produces identical arrays."""
    arr1 = cp.random.randn((5, 5), seed=42)
    arr2 = cp.random.randn((5, 5), seed=42)

    np.testing.assert_array_equal(arr1.to_list(), arr2.to_list())


def test_randn_distribution():
    """Test randn() roughly approximates standard normal characteristics."""
    # With a decent sample size, mean should be close to 0 and std close to 1
    arr = cp.random.randn(10000)
    data = np.array(arr.to_list())

    assert np.abs(np.mean(data)) < 0.1
    assert np.abs(np.std(data) - 1.0) < 0.1


def test_randn_algo_xoshiro():
    """Test randn() works with the xoshiro algorithm."""
    arr = cp.random.randn((10,), seed=123, algo="xoshiro")
    assert arr.shape == (10,)
    assert arr.dtype == cp.float32


def test_missing_backend(monkeypatch):
    """Test ImportError when rust backend is missing."""
    import corepy.random as random_module

    # Temporarily remove backend reference to trigger the exception block
    monkeypatch.setattr(random_module, "_corepy_rust", None)

    with pytest.raises(ImportError):
        random_module.rand(5)

    with pytest.raises(ImportError):
        random_module.randn(5)
