import sys

import corepy


def verify_api():
    expected_attrs = [
        "ndarray",
        "DataType",
        "array",
        "zeros",
        "ones",
        "full",
        "empty",
        "arange",
        "linspace",
        "rand",
        "randn",
        "concatenate",
        "stack",
        "split",
        "squeeze",
        "reshape",
        "flatten",
        "ravel",
        "transpose",
        "tile",
        "repeat",
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "mod",
        "floor_divide",
        "equal",
        "not_equal",
        "greater",
        "less",
        "greater_equal",
        "less_equal",
        "logical_and",
        "logical_or",
        "logical_not",
        "logical_xor",
        "sort",
        "argsort",
        "stable_sort",
        "argmax",
        "argmin",
        "where",
        "searchsorted",
        "minimum",
        "maximum",
        "take",
        "boolean_index",
        "is_even",
        "is_odd",
        "abs",
        "matmul",
        "dot",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "arctan2",
        "sinh",
        "cosh",
        "tanh",
        "arcsinh",
        "arccosh",
        "arctanh",
        "degrees",
        "radians",
        "ProfileContext",
        "profile_report",
        "DataFrame",
        "Series",
        "read_csv",
        "linalg",
        "get_backend_policy",
        "set_backend_policy",
        "get_device_info",
    ]

    missing = []
    for attr in expected_attrs:
        if not hasattr(corepy, attr):
            missing.append(attr)

    if missing:
        print(f"FAILED: Missing attributes: {', '.join(missing)}")
        sys.exit(1)

    # Check for Rust extension
    try:
        from corepy import _corepy_rust

        print("SUCCESS: _corepy_rust found")
    except ImportError:
        print("FAILED: _corepy_rust NOT found")
        sys.exit(1)

    print("VERIFICATION SUCCESS: All expected APIs are present.")


if __name__ == "__main__":
    verify_api()
