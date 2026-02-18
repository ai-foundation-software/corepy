import threading

import numpy as np
import pytest

from corepy.buffer_pool import BufferPool, get_buffer_pool, reset_buffer_pool


@pytest.fixture
def clean_pool():
    reset_buffer_pool()
    pool = BufferPool(max_size_per_class=2)  # Small size for testing eviction
    yield pool
    reset_buffer_pool()


def test_buffer_pool_lifecycle(clean_pool):
    pool = clean_pool
    size = 10
    dtype = np.float32

    # Get from empty pool -> None
    buf = pool.get(size, dtype)
    assert buf is None
    assert pool.stats()["misses"] == 1

    # Allocate (should call get internally then create new)
    # But wait, allocate method:
    # 1. get() -> None
    # 2. np.empty()
    # It does NOT automatically add to pool. release() does that.

    # Create a buffer manually
    buf1 = np.empty(size, dtype=dtype)

    # Release to pool
    pool.release(buf1, size, dtype)

    # Get from pool -> Should get buf1 back
    buf2 = pool.get(size, dtype)
    assert buf2 is buf1
    assert pool.stats()["hits"] == 1

    # Get again -> None (pool empty)
    buf3 = pool.get(size, dtype)
    assert buf3 is None


def test_eviction_policy(clean_pool):
    pool = clean_pool  # max_size = 2
    size = 10
    dtype = np.float32

    b1 = np.empty(size, dtype=dtype)
    b2 = np.empty(size, dtype=dtype)
    b3 = np.empty(size, dtype=dtype)

    # Fill pool
    pool.release(b1, size, dtype)
    pool.release(b2, size, dtype)

    # Pool has [b1, b2] (assuming append adds to right)

    # Add 3rd -> Should evict b1 (FIFO eviction as per code comments "popleft")
    pool.release(b3, size, dtype)

    assert pool.stats()["evictions"] == 1

    # Verify contents: should have b2 and b3
    # get() pops from right (LIFO). So get() should return b3, then b2.

    r1 = pool.get(size, dtype)
    assert r1 is b3

    r2 = pool.get(size, dtype)
    assert r2 is b2

    r3 = pool.get(size, dtype)
    assert r3 is None


def test_allocate_method(clean_pool):
    pool = clean_pool
    size = 20

    # 1. Allocate new
    b1 = pool.allocate(size)
    assert b1.size == size
    assert b1.dtype == np.float32

    # 2. Return it
    pool.release(b1, size, np.float32)

    # 3. Allocate again -> reuse
    b2 = pool.allocate(size)
    assert b2 is b1
    assert pool.stats()["hits"] == 1


def test_clear_and_stats(clean_pool):
    pool = clean_pool
    b = np.empty(10, dtype=np.float32)
    pool.release(b, 10, np.float32)

    # Stats
    stats = pool.stats()
    assert (
        stats["total_requests"] == 0
    )  # get() was not called, release() doesn't inc requests

    pool.get(10, np.float32)  # Hit
    stats = pool.stats()
    assert stats["hits"] == 1

    pool.release(b, 10, np.float32)

    # Memory usage
    mem = pool.memory_usage()
    assert mem["cpu_bytes"] == 10 * 4

    # Clear
    pool.clear()
    mem = pool.memory_usage()
    assert mem["cpu_bytes"] == 0


def test_singleton_access():
    reset_buffer_pool()
    p1 = get_buffer_pool()
    p2 = get_buffer_pool()
    assert p1 is p2

    reset_buffer_pool()
    p3 = get_buffer_pool()
    assert p3 is not p1


def test_gpu_pool(clean_pool):
    pool = clean_pool
    size = 5
    dtype = np.float32
    device = "metal"

    b = np.empty(size, dtype=dtype)  # Mock GPU buffer as numpy array

    pool.release(b, size, dtype, device=device)

    # Should not be in CPU pool
    assert pool.get(size, dtype, device="cpu") is None

    # Should be in GPU pool
    # Check stats BEFORE getting it back (which pops it)
    mem = pool.memory_usage()
    assert mem["gpu_bytes"] == size * 4
    assert mem["cpu_bytes"] == 0

    ret = pool.get(size, dtype, device=device)
    assert ret is b

    # After get, it's removed
    mem = pool.memory_usage()
    assert mem["gpu_bytes"] == 0
