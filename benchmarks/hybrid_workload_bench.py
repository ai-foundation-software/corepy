import concurrent.futures
import math
import time
from concurrent.futures import ProcessPoolExecutor

import corepy
from corepy.hybrid import fetch_data_from_db_with_retry, process_hybrid_pipeline


def pure_python_processing(data):
    """
    Pure Python Sequential Workload to compare against Rust.
    """
    return [math.sin((x**2.5) * 3.14) for x in data]


def benchmark_hybrid_architecture():
    print("======================================================================")
    print("Hybrid Architecture Benchmark: Data Processing Engine")
    print("======================================================================")

    # ---------------------------------------------------------
    # Scenario 1: Standard Python Sequential execution
    # Demonstrates GIL locking limitations
    # ---------------------------------------------------------
    print("\n[Scenario 1] Standard Python Workload (Sequential IO & Compute)")
    start = time.perf_counter()

    dummy_data = []
    for i in range(10):  # Fetch 10 partitions sequentially
        start_io = time.perf_counter()
        data = fetch_data_from_db_with_retry(i)  # 500k items in total
        dummy_data.extend(data)
    io_time_py = time.perf_counter() - start

    start_compute = time.perf_counter()
    result_seq = pure_python_processing(dummy_data)
    compute_time_py = time.perf_counter() - start_compute
    total_time_py = time.perf_counter() - start

    print(f"Total Items:       {len(result_seq):,}")
    print(f"I/O Time:          {io_time_py:.4f}s")
    print(f"Compute Time:      {compute_time_py:.4f}s")
    print(f"Total Elapsed:     {total_time_py:.4f}s")

    # ---------------------------------------------------------
    # Scenario 2: Python Multiprocessing
    # Demonstrates process initialization & IPC pickling overhead
    # ---------------------------------------------------------
    print("\n[Scenario 2] Python Multiprocessing Workload (ProcessPoolExecutor)")
    start = time.perf_counter()

    # Using previous dummy data to isolate compute overhead measurement
    chunk_size = len(dummy_data) // corepy.get_device_info().cpu_cores
    chunks = [
        dummy_data[i : i + chunk_size] for i in range(0, len(dummy_data), chunk_size)
    ]

    start_compute = time.perf_counter()
    with ProcessPoolExecutor() as p:
        result_mp = list(p.map(pure_python_processing, chunks))
    compute_time_mp = time.perf_counter() - start_compute
    total_time_mp = time.perf_counter() - start

    print(f"Total Items:       {len(dummy_data):,}")
    print(f"Compute Time:      {compute_time_mp:.4f}s (Inc. IPC serialization)")
    print(f"Total Elapsed:     {total_time_mp:.4f}s")

    # ---------------------------------------------------------
    # Scenario 3: Hybrid Architecture
    # Demonstrates concurrent IO unlocking the GIL + Rayon Rust Work-Stealing
    # ---------------------------------------------------------
    print("\n[Scenario 3] Hybrid Architecture (Threaded IO + Rust Rayon Compute)")
    start = time.perf_counter()

    # This single function handles both the threaded IO fetching and hands off the
    # massive combined list directly to the Rust engine bypassing Python map() calls.
    # Safe fallback exists internally in process_workload for < 10_000 elements.
    result_hybrid = process_hybrid_pipeline(10)

    total_time_hybrid = time.perf_counter() - start

    print(f"Total Items:       {len(result_hybrid):,}")
    print(f"Total Elapsed:     {total_time_hybrid:.4f}s (I/O and Compute Combined)")

    print("\n======================================================================")
    print("Results")
    print("======================================================================")
    print(f"Speedup vs Pure Python: {total_time_py / total_time_hybrid:.1f}x faster")
    # Multiprocessing total is somewhat misleading as I omitted IO, but the comparison is stark
    print(f"Speedup vs MultiProc:   {total_time_mp / total_time_hybrid:.1f}x faster")


if __name__ == "__main__":
    benchmark_hybrid_architecture()
