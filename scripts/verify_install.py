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
        print("✅ Matmul test passed")
        print(f"✅ Dispatch: {corepy.explain_last_dispatch()}")
    except Exception as e:
        print(f"❌ Matmul test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
