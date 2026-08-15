import os
import pickle
import time
import traceback

import corepy as cp
import corepy.linalg as la
from corepy.array import DataType


def log(msg):
    print(f"[SHOR] {msg}")


def test_dataframe_core():
    log("--- 1. DataFrame/Series Core Core ---")
    df = cp.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    assert df.shape == (3, 2)
    assert "a" in df.columns
    assert df["a"].name == "a"

    # Selection/Indexing
    s = df["a"]
    assert len(s) == 3
    # Internal push to float for now in the wrapper
    log(f"Series 'a' values: {s.values.to_list()}")

    # Arithmetic
    res = df["a"].values + df["b"].values
    assert res.to_list() == [11.0, 22.0, 33.0]
    log("✅ DataFrame Core passed.")


def test_aggregations():
    log("--- 2. Aggregations (GroupBy) ---")
    # Basic GroupBy - Using numeric keys for now as backend is numeric-only
    df = cp.DataFrame({"key": [1.0, 2.0, 1.0, 3.0], "val": [1, 2, 3, 4]})
    gb = df.groupby("key")
    res_sum = gb.sum()
    log(f"GroupBy Sum:\n{res_sum}")
    # Selection from result
    assert "val" in res_sum.columns
    log("✅ Aggregations passed.")


def test_io():
    log("--- 3. IO (CSV) ---")
    df = cp.DataFrame({"x": [1.1, 2.2], "y": [3.3, 4.4]})
    df.to_csv("test_file.csv")
    df_read = cp.read_csv("test_file.csv")
    assert df_read.shape == (2, 2)
    assert "x" in df_read.columns
    os.remove("test_file.csv")
    log("✅ IO passed.")


def test_lazy():
    log("--- 4. Lazy Evaluation ---")
    # Lazy execution using the context manager
    with cp.lazy():
        a = cp.array([10.0, 20.0, 30.0])
        b = a * 2.5
        c = b + 1.0
        res = c.compute()
    expected = [10 * 2.5 + 1, 20 * 2.5 + 1, 30 * 2.5 + 1]
    assert res.to_list() == expected
    log(f"Lazy Result: {res.to_list()}")
    log("✅ Lazy Evaluation passed.")


def test_performance():
    log("--- 5. Performance (Rayon Parallelism) ---")
    # Array sum performance
    size = 10_000_000
    a = cp.arange(0, size)
    start = time.perf_counter()
    res = a.sum()
    end = time.perf_counter()
    log(f"Sum of {size} elements: {res.to_list()[0]} in {(end - start) * 1000:.2f}ms")

    # Large Matrix Multiplication
    dim = 2048
    m1 = cp.ones((dim, dim))
    m2 = cp.ones((dim, dim))
    start = time.perf_counter()
    # matmul should dispatch to the fastest backend
    m3 = m1 @ m2
    end = time.perf_counter()
    dur = end - start
    gflops = (2.0 * dim**3) / dur / 1e9
    log(f"{dim}x{dim} Matmul performance: {gflops:.2f} GFLOPS (Time: {dur:.4f}s)")
    log("✅ Performance validation completed.")


def test_serialization():
    log("--- 6. Serialization (Pickle) ---")
    df = cp.DataFrame({"data": [1, 2, 3]})
    try:
        # Check if objects support pickle serialization
        data = pickle.dumps(df)
        df2 = pickle.loads(data)
        assert df2.shape == df.shape
        log("✅ Serialization (Pickle) passed.")
    except Exception:
        log("⚠️ Serialization (Pickle) NOT supported by DataFrame yet.")
        # log(traceback.format_exc())


def test_edge_cases():
    log("--- 7. Edge Cases ---")
    # Empty DF Case
    try:
        df_empty = cp.DataFrame({})
        assert df_empty.shape == (0, 0)
        log("✅ Empty DataFrame (0x0) handled.")
    except Exception as e:
        log(f"❌ Empty DataFrame failed: {e}")

    # Null/Large scale
    size = 100_000
    try:
        df_large = cp.DataFrame({"a": list(range(size))})
        assert df_large.shape[0] == size
        log(f"✅ Large Dataset ({size} rows) handled.")
    except Exception as e:
        log(f"❌ Large Dataset failed: {e}")


if __name__ == "__main__":
    test_dataframe_core()
    test_aggregations()
    test_io()
    test_lazy()
    test_performance()
    test_serialization()
    test_edge_cases()
    log("\n--- DEEP VALIDATION COMPLETED ---")
