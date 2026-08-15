"""
Tutorial 10: Advanced Memory Management
=======================================
This tutorial showcases the built-in Buffer Pool memory manager.
When executing many temporary operations, constantly allocating and
deallocating memory from the OS is a huge bottleneck. CorePy reuses
allocations automatically with an LRU cache.
"""

import corepy as cp
from corepy.buffer_pool import get_buffer_pool


def main():
    print("--- Memory Allocation & Buffer Pooling ---")
    pool = get_buffer_pool()

    print("Initial Pool Stats:")
    print(pool.stats())

    # Let's perform a few operations that allocate and free memory heavily
    # This loop simulates processing many distinct batches of data
    for _ in range(100):
        # We assume these operations internally fetch and release from the pool
        # For tutorial purposes we'll interact directly with the pool as a demo
        buf = pool.get(size=1024, dtype="float32", device="cpu")
        if buf is None:
            buf = pool.allocate(size=1024, dtype="float32", device="cpu")

        # ... Do work with buf ...

        # Release back to pool
        pool.release(buf, size=1024, dtype="float32", device="cpu")

    print("\\nPool Stats After 100 Operations:")
    stats = pool.stats()
    print(f"Hits:      {stats['hits']}")
    print(f"Misses:    {stats['misses']} (Allocated new)")
    print(f"Evictions: {stats['evictions']}")
    print(f"Hit Rate:  {stats['hit_rate']:.2f}%")

    mem_usage = pool.memory_usage()
    print(f"Total Cached Bytes: {mem_usage['cpu_bytes']}")


if __name__ == "__main__":
    main()
