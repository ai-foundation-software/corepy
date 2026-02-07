"""
Tutorial 05: Profiler & Export
==============================

This tutorial showcases the new Profiler Export features in Corepy v0.2.3.
You can now export performance traces to Google Chrome's tracing tool.

Steps:
1. Enable profiling
2. Run complex operations
3. Export to JSON
4. Visualize in chrome://tracing or ui.perfetto.dev
"""

import time

import numpy as np

import corepy as cp


def heavy_computation():
    # Simulate a workload
    N = 1000
    # Create shaped data using NumPy first
    a_np = np.full((N, N), 1.0, dtype=np.float32)
    b_np = np.full((N, N), 2.0, dtype=np.float32)

    a = cp.Tensor(a_np)
    b = cp.Tensor(b_np)

    # Matmul is heavy
    c = a @ b

    # Element-wise is bound by memory
    d = c + a
    e = d / 2.0

    return e.sum()


def main():
    print("=== Corepy Profiler Export Tutorial ===")

    # 1. Enable Profiling
    print("Enable profiling...")
    cp.enable_profiling()

    # 2. Run Workload
    print("Running computation...")
    for i in range(3):
        print(f"  Iteration {i + 1}...")
        heavy_computation()

    # 3. Report to Console
    print("\n--- Console Summary ---")
    print(cp.profile_report())

    # 4. Export Trace
    output_file = "corepy_trace.json"
    print(f"\n--- Exporting Trace to {output_file} ---")

    # This API is new in v0.2.3
    try:
        path = cp.profiler.export_chrome_trace(output_file)
        print(f"✅ Trace exported successfully to: {path}")
        print("\nTo visualize:")
        print("1. Open Chrome browser")
        print("2. Navigate to chrome://tracing")
        print("3. Click 'Load' and select the file.")
        print("OR visit https://ui.perfetto.dev/ and drop the file there.")
    except AttributeError:
        print("❌ `export_chrome_trace` not found. Are you on v0.2.3?")
    except Exception as e:
        print(f"❌ Export failed: {e}")


if __name__ == "__main__":
    main()
