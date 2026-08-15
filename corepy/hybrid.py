import os
import time
from concurrent.futures import ThreadPoolExecutor

from corepy import _corepy_rust  # The compiled Rust extension

# IO heavy tasks get bounded threads proportional to CPU cores.
# We limit thread creation to prevent OS swapping while allowing enough concurrency to saturate network/disk pipes.
MAX_IO_THREADS = min(32, (os.cpu_count() or 1) * 4)
io_pool = ThreadPoolExecutor(max_workers=MAX_IO_THREADS)


def fetch_data_from_db_with_retry(partition_id: int, retries: int = 3) -> list:
    """Simulates fetching DB rows with graceful error handling and retry logic for IO Tasks."""
    for attempt in range(retries):
        try:
            # Simulate network IO latency
            time.sleep(0.05)
            # Fetch dummy vector data simulating a DB response
            return [1.0] * 50_000
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"Task failed after {retries} attempts: {e}"
                ) from None
            time.sleep(2**attempt)  # Exponential backoff


def process_hybrid_pipeline(num_partitions: int):
    """
    Executes the Hybrid Architecture processing pipeline.
    1. Python ThreadPool instances manage IO/Network fetching independently from the GIL lock.
    2. Python orchestrator collates the data.
    3. Rust Engine handles the pure computation.
    """
    # 1. IO Heavy -> Python threads
    futures = []
    for i in range(num_partitions):
        futures.append(io_pool.submit(fetch_data_from_db_with_retry, i))

    # Collate results
    combined_data = []
    for f in futures:
        combined_data.extend(f.result())

    # 2. CPU Heavy -> Rust Engine (Bypasses Python completely for raw speed)
    return _corepy_rust.process_workload(combined_data)
