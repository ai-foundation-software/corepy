"""
Tutorial 14: High-Performance Hybrid Architecture
=================================================

This tutorial demonstrates the architectural principles for creating scalable,
mission-critical data processing engines leveraging Python for IO orchestration
and Rust for raw parallel compute.

Running this design ensures your application achieves C++ level performance while
maintaining the rapid development cycle, massive ecosystem, and API elegance of Python.

Core Principles Demonstrated:
1. Segregation of Duties:
    - IO/Networking: Handled by Python `ThreadPoolExecutor` (GIL is released during IO)
    - CPU/Crunching: Handled by compiled Rust Extension (Rayon)

2. Adaptive Performance (The Parallel Threshold Rule):
    - Spinning up OS threads introduces context switching latency.
    - If the user sends fewer than 10,000 items into the CorePy engine, it bypasses
      Rayon completely and automatically executes sequentially to optimize microseconds.
    - If data is large, it scales dynamically via Rayon Work-Stealing.

3. System Safety Margin:
    - Never consume 100% of logical CPUs blindly.
    - `corepy [rust]` automatically initializes `usable_threads = max(1, cpu_threads - 1)`.
    - This reserves one core for the host OS to handle DB queries, UI polling, docker
      management, and the Python orchestrator itself, preventing complete system freezing.
"""

import math
import time

import corepy
from corepy.hybrid import fetch_data_from_db_with_retry, process_hybrid_pipeline


def main():
    print("===========================================")
    print("  System Design Principles Execution Demo  ")
    print("===========================================")

    device_info = corepy.get_device_info()
    print(f"Detected System Threads: {device_info.cpu_threads}")
    print(f"Allocated Rayon Threads: {max(1, device_info.cpu_threads - 1)}\n")

    # -------------------------------------------------------------
    # 1. The Threshold Fallback Rule
    # -------------------------------------------------------------
    print("1. Executing small data batch (Threshold Fallback Rule)...")
    small_batch = [float(i) for i in range(5_000)]  # Below 10,000 threshold

    # Executes via pure serial sequence inside the Rust core to save thread spawning latency
    start = time.perf_counter()
    _ = corepy._corepy_rust.process_workload(small_batch)
    print(
        f"Completed smoothly without Rayon spinup in {time.perf_counter() - start:.5f}s\n"
    )

    # -------------------------------------------------------------
    # 2. Hybrid Orchestration (Concurrency Demo)
    # -------------------------------------------------------------
    print("2. Executing massive hybrid orchestrator pipeline...")
    # This will kick off Python ThreadPools to fetch 500k "DB Rows" via the network,
    # wait on non-blocking IO handles, collate them seamlessly, and dispatch the entire
    # array into Rust where Rayon will work-steal chunks across X-1 logical cores.
    start = time.perf_counter()
    result = process_hybrid_pipeline(num_partitions=10)  # 500,000 items total
    elapsed = time.perf_counter() - start

    print(f"Total processed elements: {len(result):,}")
    print(f"Total elapsed time (IO + Compute): {elapsed:.4f}s")
    print(
        "\nMission Accomplished: High-performance, memory-safe, failure-resistant pipeline completed."
    )


if __name__ == "__main__":
    main()
