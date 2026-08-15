import json

import pytest

import corepy as cp
from corepy.profiler import (
    ProfileContext,
    clear_profile,
    detect_bottlenecks,
    disable_profiling,
    enable_profiling,
    profile_operation,
    profile_report,
)


@pytest.fixture(autouse=True)
def clean_profiler():
    """Reset profiler before and after each test."""
    clear_profile()
    disable_profiling()
    yield
    clear_profile()
    disable_profiling()


def test_enable_disable():
    """Test enabling and disabling the profiler."""
    # Should start disabled
    t1 = cp.ndarray([1.0, 2.0])
    _ = t1 + t1
    report = profile_report(format="dict")
    assert len(report.get("operations", {})) == 0

    # Enable
    enable_profiling()
    t1 = cp.ndarray([1.0, 2.0])
    _ = t1 + t1
    report = profile_report(format="dict")
    ops = report.get("operations", {})
    assert "add" in ops
    assert ops["add"]["count"] >= 1

    # Disable
    disable_profiling()
    count_before = ops["add"]["count"]
    _ = t1 + t1
    report = profile_report(format="dict")
    ops = report.get("operations", {})
    # Count should not increase after disable
    assert ops["add"]["count"] == count_before


def test_context_manager():
    """Test scoping via context manager."""
    enable_profiling()

    t1 = cp.ndarray([1.0, 2.0])

    with ProfileContext("scope_a"):
        _ = t1 + t1

    with ProfileContext("scope_b"):
        _ = t1 * t1

    report = profile_report(format="dict")
    # Use context filtering (if supported by Rust, otherwise check for events)
    # Current implementation might not filter by context in 'get_profile_report' params efficiently yet,
    # but let's check general behavior.

    # Verify we have ops
    assert "add" in report["operations"]
    assert "mul" in report["operations"]


def test_decorator():
    """Test @profile_operation decorator."""
    enable_profiling()

    @profile_operation
    def my_custom_op(x):
        return x + x

    t = cp.ndarray([10.0])
    _ = my_custom_op(t)

    report = profile_report(format="dict")
    assert "add" in report["operations"]


def test_report_formats():
    """Test different report export formats."""
    enable_profiling()
    t = cp.ndarray([1.0])
    _ = t + t

    # JSON String
    json_str = profile_report(format="json")
    assert isinstance(json_str, str)
    assert "operations" in json.loads(json_str)

    # Dict
    data = profile_report(format="dict")
    assert isinstance(data, dict)
    assert "add" in data["operations"]

    # Table (Text)
    text = profile_report(format="table")
    assert isinstance(text, str)
    assert "COREPY PROFILE REPORT" in text
    assert "add" in text


def test_metrics_accuracy():
    """Test that metrics capture counts correctly."""
    enable_profiling()
    t = cp.ndarray([1.0])

    loops = 10
    for _ in range(loops):
        _ = t + t  # add

    report = profile_report(format="dict")
    add_ops = report["operations"]["add"]

    # Depending on implementation, __add__ might dispatch multiple events
    # if it calls other helpers, but for now we expect at least 'loops' counts.
    assert add_ops["count"] == loops
    assert add_ops["total_time_ms"] >= 0.0


def test_matmul_profiling():
    """Test that matmul (FFI dispatch) is profiled."""
    enable_profiling()
    t = cp.ndarray([1.0, 2.0])

    # Dot product (matmul of 1D)
    _ = t.matmul(t)

    report = profile_report(format="dict")
    ops = report["operations"]

    assert ("matmul" in ops) or ("dot_product" in ops)
    op_key = "matmul" if "matmul" in ops else "dot_product"
    assert ops[op_key]["count"] == 1
    assert ops[op_key]["primary_backend"] == "CPU"


def test_bottleneck_detection():
    """Test bottleneck detection logic."""
    enable_profiling()
    # Force enable python profiler because enable_profiling() calls Rust one (bound at import)
    cp.profiler.core._python_profiler.enable()
    from unittest.mock import patch

    from corepy.profiler.core import record_op

    # Manual override because patch seems to fail on bound C-extension function variable
    original_get_report = cp.profiler.core._get_profile_report
    cp.profiler.core._get_profile_report = cp.profiler.core._python_profiler.get_report

    try:
        with patch("corepy.profiler.core._RUST_AVAILABLE", False):
            for _ in range(100):
                record_op("mock_sum_op", 1.0, "CPU")

        bottlenecks = detect_bottlenecks(threshold=0.0)

    finally:
        # Restore
        cp.profiler.core._get_profile_report = original_get_report

    found = any(b["operation"] == "mock_sum_op" for b in bottlenecks)
    assert found, "mock_sum_op operation not detected in bottlenecks"

    found_sum = next(b for b in bottlenecks if b["operation"] == "mock_sum_op")
    assert found_sum["percent_total"] > 90.0
