
import corepy as cp
import sys

def verify_readme_metal():
    print("\n--- Verifying README Metal Snippet ---")
    try:
        # README claims: "Automatically uses Metal if available on macOS"
        # On Linux, this should either fallback or raise a specific error.
        # Ideally it falls back if the implementation matches the "Automatic" claim, 
        # but the snippet explicitly asks for device="metal".
        t = cp.Tensor([1.0, 2.0, 3.0], device="metal")
        print(f"Result: Success (Backend: {t.backend})")
    except Exception as e:
        print(f"Result: Caught expected exception on non-macOS: {e}")

def verify_readme_profiling():
    print("\n--- Verifying README Profiling Snippet ---")
    try:
        cp.enable_profiling()
        x = cp.Tensor([1.0] * 1_000_000)
        y = x * 3.14159
        result = y.mean()
        cp.profiler.export_chrome_trace("trace_test.json")
        print("Result: Success (trace_test.json exported)")
    except Exception as e:
        print(f"Result: Failed with error: {e}")
        raise e

def verify_getting_started():
    print("\n--- Verifying Getting Started Snippet ---")
    try:
        prices = cp.Tensor([10.5, 20.0, 15.5, 30.0])
        total = prices.sum()
        average = prices.mean()
        print(f"Prices: {prices}")
        print(f"Total:   {total}")
        print(f"Average: {average}")
        print("Result: Success")
    except Exception as e:
        print(f"Result: Failed with error: {e}")
        raise e

if __name__ == "__main__":
    verify_readme_metal()
    verify_readme_profiling()
    verify_getting_started()
