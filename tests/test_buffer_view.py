"""Unit tests for BufferView abstraction (Phase 5.1)."""

import numpy as np
import pytest

from corepy.buffer import (
    BufferView,
    CPU,
    Device,
    DeviceType,
    MemoryType,
    from_buffer,
    from_numpy,
)
from corepy.backend.types import DataType


class TestDeviceType:
    """Tests for DeviceType enum."""

    def test_cpu_type(self):
        assert DeviceType.CPU.value == "cpu"

    def test_cuda_type(self):
        assert DeviceType.CUDA.value == "cuda"


class TestDevice:
    """Tests for Device abstraction."""

    def test_default_cpu(self):
        device = Device(DeviceType.CPU)
        assert device.is_cpu()
        assert not device.is_cuda()
        assert str(device) == "cpu"

    def test_cuda_device(self):
        device = Device(DeviceType.CUDA, 0)
        assert not device.is_cpu()
        assert device.is_cuda()
        assert str(device) == "cuda:0"

    def test_cuda_multi_gpu(self):
        device = Device(DeviceType.CUDA, 1)
        assert str(device) == "cuda:1"

    def test_device_equality(self):
        d1 = Device(DeviceType.CPU)
        d2 = Device(DeviceType.CPU)
        d3 = Device(DeviceType.CUDA, 0)
        assert d1 == d2
        assert d1 != d3

    def test_cpu_singleton(self):
        assert CPU.is_cpu()
        assert str(CPU) == "cpu"


class TestBufferView:
    """Tests for BufferView dataclass."""

    def test_from_numpy_contiguous(self):
        """Zero-copy path for contiguous arrays."""
        arr = np.arange(10, dtype=np.float32)
        view = from_numpy(arr, device=CPU, dtype=DataType.FLOAT32)

        assert view.data_ptr == arr.__array_interface__["data"][0]
        assert view.shape == (10,)
        assert view.strides is None  # Contiguous
        assert view.is_contiguous()
        assert view.device == CPU
        assert view.owner is arr

    def test_from_numpy_sliced(self):
        """Detects non-contiguous arrays."""
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        sliced = arr[::2, ::2]  # Non-contiguous!

        view = from_numpy(sliced, device=CPU, dtype=DataType.FLOAT32)

        assert view.strides is not None
        assert not view.is_contiguous()
        assert view.shape == (5, 5)

    def test_is_contiguous_explicit_strides(self):
        """Test stride validation with various layouts."""
        # 2D array 3x4 float32
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        view = from_numpy(arr, device=CPU, dtype=DataType.FLOAT32)

        # Contiguous strides for 3x4 float32: (16, 4)
        assert view.is_contiguous()

    def test_ensure_contiguous_no_copy_needed(self):
        """Already contiguous - should return self."""
        arr = np.arange(10, dtype=np.float32)
        view = from_numpy(arr)

        result = view.ensure_contiguous()
        assert result is view  # Same object

    def test_ensure_contiguous_copies_when_needed(self, caplog):
        """Non-contiguous array triggers copy with warning."""
        import logging

        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        sliced = arr[::2, ::2]
        view = from_numpy(sliced)

        assert not view.is_contiguous()

        with caplog.at_level(logging.WARNING):
            contiguous = view.ensure_contiguous()

        assert contiguous.is_contiguous()
        assert contiguous is not view
        assert "non-contiguous" in caplog.text.lower()

    def test_device_defaults_to_cpu(self):
        """Default device should be CPU."""
        arr = np.arange(10, dtype=np.float32)
        view = from_numpy(arr)  # No device specified

        assert view.device == CPU
        assert view.device.is_cpu()

    def test_element_count(self):
        """Test element count property."""
        arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        view = from_numpy(arr)

        assert view.element_count == 24

    def test_nbytes(self):
        """Test nbytes property."""
        arr = np.arange(10, dtype=np.float32)  # 10 * 4 bytes
        view = from_numpy(arr, dtype=DataType.FLOAT32)

        assert view.nbytes == 40

    def test_writable_flag(self):
        """Test writable flag extraction."""
        arr = np.arange(10, dtype=np.float32)
        view = from_numpy(arr)
        assert view.writable

        arr.flags.writeable = False
        view_readonly = from_numpy(arr)
        assert not view_readonly.writable


class TestFromBuffer:
    """Tests for from_buffer factory function."""

    def test_from_bytes(self):
        """Create BufferView from bytes."""
        data = bytes([1, 2, 3, 4])
        view = from_buffer(data, dtype=DataType.INT32, shape=(1,))

        assert view.shape == (1,)
        assert view.device == CPU
        assert not view.writable  # bytes are immutable

    def test_from_bytearray(self):
        """Create BufferView from bytearray."""
        data = bytearray([1, 2, 3, 4])
        view = from_buffer(data, dtype=DataType.INT32, shape=(1,))

        assert view.writable


class TestDataTypeItemsize:
    """Tests for DataType.itemsize property."""

    def test_float32_size(self):
        assert DataType.FLOAT32.itemsize == 4

    def test_float64_size(self):
        assert DataType.FLOAT64.itemsize == 8

    def test_int32_size(self):
        assert DataType.INT32.itemsize == 4

    def test_int64_size(self):
        assert DataType.INT64.itemsize == 8

    def test_bool_size(self):
        assert DataType.BOOL.itemsize == 1
