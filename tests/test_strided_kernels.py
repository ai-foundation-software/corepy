"""Unit tests for strided kernels (Phase 5.2)."""

import logging

import numpy as np
import pytest

from corepy import Tensor
from corepy.backend.types import DataType


class TestStridedSum:
    """Tests for strided sum operations."""

    def test_sum_contiguous(self):
        """Contiguous array uses fast path."""
        arr = np.arange(100, dtype=np.float32)
        t = Tensor(arr)
        result = t.sum()
        
        expected = arr.sum()
        # Extract scalar from tensor
        actual = result._backing_data[0]
        
        assert abs(actual - expected) < 1e-4

    def test_sum_strided_2d(self):
        """Sliced 2D array uses strided kernel (zero-copy)."""
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        sliced = arr[::2, ::2]  # Non-contiguous: 5x5 view
        
        t = Tensor(sliced)
        result = t.sum()
        
        expected = sliced.sum()
        actual = result._backing_data[0]
        
        assert abs(actual - expected) < 1e-4

    def test_sum_strided_3d(self):
        """Sliced 3D array uses strided kernel."""
        arr = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
        sliced = arr[::2, ::2, ::2]  # 5x5x5 view
        
        t = Tensor(sliced)
        result = t.sum()
        
        expected = sliced.sum()
        actual = result._backing_data[0]
        
        assert abs(actual - expected) < 1e-3

    def test_sum_strided_no_warning(self, caplog):
        """Strided sum should not produce copy warning."""
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        sliced = arr[::2, ::2]
        
        t = Tensor(sliced)
        
        with caplog.at_level(logging.WARNING):
            _ = t.sum()
        
        # Should NOT contain copy warning (zero-copy path used)
        assert "non-contiguous" not in caplog.text.lower()


class TestStridedMean:
    """Tests for strided mean operations."""

    def test_mean_contiguous(self):
        """Contiguous array uses fast path."""
        arr = np.arange(100, dtype=np.float32)
        t = Tensor(arr)
        result = t.mean()
        
        expected = arr.mean()
        actual = result._backing_data[0]
        
        assert abs(actual - expected) < 1e-4

    def test_mean_strided_2d(self):
        """Sliced 2D array uses strided kernel."""
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        sliced = arr[::2, ::2]
        
        t = Tensor(sliced)
        result = t.mean()
        
        expected = sliced.mean()
        actual = result._backing_data[0]
        
        assert abs(actual - expected) < 1e-4


class TestCorrectnessVsNumpy:
    """Compare strided results with NumPy for correctness."""

    @pytest.mark.parametrize("shape,slice_spec", [
        ((20, 20), (slice(None, None, 2), slice(None, None, 2))),
        ((10, 10, 10), (slice(None, None, 3), slice(None, None, 2), slice(None, None, 2))),
        ((100,), (slice(None, None, 5),)),
    ])
    def test_sum_matches_numpy(self, shape, slice_spec):
        """Strided sum matches NumPy result."""
        arr = np.random.randn(*shape).astype(np.float32)
        sliced = arr[slice_spec]
        
        t = Tensor(sliced)
        result = t.sum()._backing_data[0]
        expected = sliced.sum()
        
        assert abs(result - expected) < 1e-3, f"Expected {expected}, got {result}"

    @pytest.mark.parametrize("shape,slice_spec", [
        ((20, 20), (slice(None, None, 2), slice(None, None, 2))),
        ((100,), (slice(None, None, 5),)),
    ])
    def test_mean_matches_numpy(self, shape, slice_spec):
        """Strided mean matches NumPy result."""
        arr = np.random.randn(*shape).astype(np.float32)
        sliced = arr[slice_spec]
        
        t = Tensor(sliced)
        result = t.mean()._backing_data[0]
        expected = sliced.mean()
        
        assert abs(result - expected) < 1e-4, f"Expected {expected}, got {result}"
