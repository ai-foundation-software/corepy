"""
Buffer Pool for Memory Management

Reuses allocated CoreArray buffers to reduce allocation/deallocation overhead.
Thread-safe implementation with LRU eviction for memory control.
"""

import threading
from collections import defaultdict, deque
from typing import Any, Dict, Optional, Tuple


class BufferPool:
    """
    Memory buffer pool for reusing allocations.

    Maintains separate pools for different sizes and devices (CPU/GPU).
    Uses LRU eviction when pools grow too large.
    """

    def __init__(self, max_size_per_class: int = 100):
        """
        Initialize buffer pool.

        Args:
            max_size_per_class: Maximum buffers to cache per (size, dtype, device) combination
        """
        self._max_size = max_size_per_class

        # Separate pools for CPU and GPU
        # Key: (size, dtype_str, device) -> deque of buffers
        self._cpu_buffers: Dict[Tuple[int, str, str], deque] = defaultdict(deque)
        self._gpu_buffers: Dict[Tuple[int, str, str], deque] = defaultdict(deque)

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(
        self, size: int, dtype: Any = "float32", device: str = "cpu"
    ) -> Optional[Any]:
        """
        Get a buffer from the pool or None if unavailable.

        Args:
            size: Number of elements needed
            dtype: Data type string
            device: Target device ('cpu', 'metal', 'cuda')

        Returns:
            Cached buffer or None if not available
        """
        dtype_str = str(dtype)
        key = (size, dtype_str, device)

        with self._lock:
            buffers = self._cpu_buffers if device == "cpu" else self._gpu_buffers

            if key in buffers and buffers[key]:
                buffer = buffers[key].pop()
                self._hits += 1
                return buffer
            else:
                self._misses += 1
                return None

    def release(
        self, buffer: Any, size: int, dtype: Any = "float32", device: str = "cpu"
    ):
        """
        Return a buffer to the pool for reuse.

        Args:
            buffer: CoreArray or list buffer to return to pool
            size: Number of elements in buffer
            dtype: Data type string
            device: Device where buffer resides
        """
        dtype_str = str(dtype)
        key = (size, dtype_str, device)

        with self._lock:
            buffers = self._cpu_buffers if device == "cpu" else self._gpu_buffers

            if len(buffers[key]) >= self._max_size:
                buffers[key].popleft()
                self._evictions += 1

            buffers[key].append(buffer)

    def allocate(self, size: int, dtype: Any = "float32", device: str = "cpu") -> Any:
        """
        Allocate a buffer, reusing from pool if available.

        Args:
            size: Number of elements needed
            dtype: Data type string (default: 'float32')
            device: Target device

        Returns:
            CoreArray or list buffer
        """
        buffer = self.get(size, dtype, device)

        if buffer is not None:
            return buffer

        # Allocate new buffer via CoreArray if available
        try:
            from ._corepy_rust import _RustCoreArray as CoreArray

            return CoreArray.zeros([size])
        except ImportError:
            return [0.0] * size

    def clear(self, device: Optional[str] = None):
        """
        Clear all buffers from the pool.

        Args:
            device: If specified, only clear buffers for this device.
                   If None, clear all buffers.
        """
        with self._lock:
            if device is None:
                self._cpu_buffers.clear()
                self._gpu_buffers.clear()
            elif device == "cpu":
                self._cpu_buffers.clear()
            else:
                self._gpu_buffers.clear()

    def stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,  # type: ignore[dict-item]
                "evictions": self._evictions,
                "total_requests": total,
            }

    def memory_usage(self) -> Dict[str, int]:
        """Estimate total memory used by pooled buffers."""
        with self._lock:
            cpu_bytes = sum(
                sum(getattr(buf, "nbytes", 0) for buf in buffers)
                for buffers in self._cpu_buffers.values()
            )

            gpu_bytes = sum(
                sum(getattr(buf, "nbytes", 0) for buf in buffers)
                for buffers in self._gpu_buffers.values()
            )

            return {
                "cpu_bytes": cpu_bytes,
                "gpu_bytes": gpu_bytes,
                "total_bytes": cpu_bytes + gpu_bytes,
                "cpu_mb": cpu_bytes / (1024 * 1024),  # type: ignore[dict-item]
                "gpu_mb": gpu_bytes / (1024 * 1024),  # type: ignore[dict-item]
            }


# Global buffer pool instance
_global_pool: Optional[BufferPool] = None
_pool_lock = threading.Lock()


def get_buffer_pool() -> BufferPool:
    """Get the global buffer pool instance (thread-safe singleton)."""
    global _global_pool

    if _global_pool is None:
        with _pool_lock:
            if _global_pool is None:
                from .config import BUFFER_POOL_MAX_SIZE

                _global_pool = BufferPool(max_size_per_class=BUFFER_POOL_MAX_SIZE)

    return _global_pool


def reset_buffer_pool():
    """Reset the global buffer pool (useful for testing)."""
    global _global_pool
    with _pool_lock:
        if _global_pool is not None:
            _global_pool.clear()
        _global_pool = None
