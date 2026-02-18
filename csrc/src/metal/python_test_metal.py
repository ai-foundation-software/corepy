# Simple Python runner for the Metal backend self-test using ctypes.
#
# Usage:
#   python3 python_test_metal.py --dylib /path/to/libYourMetal.dylib \
#       [--metallib /path/to/kernels.metallib]
#
# Notes:
# - If your project builds a different library name or location, pass it via --dylib.
# - Optionally provide --metallib to set COREPY_METAL_LIB_PATH for library loading.
# - The function metal_self_test() must be exported with C linkage (already the case).

import argparse
import ctypes
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run Metal backend self-test via ctypes"
    )
    parser.add_argument(
        "--dylib", required=True, help="Path to the built dynamic library (.dylib)"
    )
    parser.add_argument(
        "--metallib", help="Path to kernels.metallib; sets COREPY_METAL_LIB_PATH"
    )
    args = parser.parse_args()

    if args.metallib:
        os.environ["COREPY_METAL_LIB_PATH"] = os.path.expanduser(args.metallib)
        print(f"[python] COREPY_METAL_LIB_PATH={os.environ['COREPY_METAL_LIB_PATH']}")

    dylib_path = os.path.expanduser(args.dylib)
    if not os.path.exists(dylib_path):
        print(f"[python] dylib not found: {dylib_path}", file=sys.stderr)
        return 2

    try:
        lib = ctypes.CDLL(dylib_path)
    except OSError as e:
        print(f"[python] Failed to load dylib: {e}", file=sys.stderr)
        return 3

    # Declare function signatures
    # int metal_self_test();
    try:
        metal_self_test = lib.metal_self_test
        metal_self_test.restype = ctypes.c_int
        metal_self_test.argtypes = []
    except AttributeError:
        print(
            "[python] metal_self_test not found in library. Ensure it's exported.",
            file=sys.stderr,
        )
        return 4

    rc = metal_self_test()
    print(f"[python] metal_self_test() returned {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
