import sys

import numpy as np

import corepy


def main():
    print("=== Verification ===")
    print(f"✅ corepy {corepy.__version__} loaded")
    print(f"✅ Backend: {corepy.get_backend_policy()}")

    # Test matmul
    try:
        a = corepy.array(np.random.randn(100, 100).astype(np.float32))
        b = corepy.array(np.random.randn(100, 100).astype(np.float32))
        c = a.matmul(b)
        print("✅ Matmul test passed (CPU)")
        print(f"✅ Dispatch: {corepy.explain_last_dispatch()}")
    except Exception as e:
        print(f"❌ Matmul test failed: {e}")
        sys.exit(1)

    # Test Metal if on macOS
    if sys.platform == "darwin":
        import platform

        if platform.machine() == "arm64":
            print("\n=== Metal Verification ===")
            try:
                # Basic Metal test
                a_metal = corepy.array([1.0, 2.0, 3.0], device="metal")
                res = a_metal.sum()
                print(f"✅ Metal array created: {a_metal}")
                print(f"✅ Metal sum result: {res}")
            except Exception as e:
                print(f"⚠️ Metal test failed or not supported: {e}")
                # Don't fail the whole script if Metal isn't available (e.g. CI without GPU)
        else:
            print("\n⚠️ Skipping Metal test (not apple silicon)")


if __name__ == "__main__":
    main()
