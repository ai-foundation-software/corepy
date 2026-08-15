import corepy as cp


def check_attrs(obj, attrs):
    missing = []
    for attr in attrs:
        try:
            has_it = hasattr(obj, attr)
            if not has_it:
                missing.append(attr)
        except Exception:
            # If accessing it throws something like NotImplementedError, it exists but is broken or fallback-only
            pass
    return missing


print("Checking API compliance...")

missing_cp = check_attrs(
    cp,
    [
        "array",
        "zeros",
        "ones",
        "full",
        "empty",
        "arange",
        "linspace",
        "eye",
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "sqrt",
        "exp",
        "log",
        "abs",
        "maximum",
        "minimum",
        "clip",
        "sum",
        "mean",
        "std",
        "var",
        "min",
        "max",
        "argmin",
        "argmax",
        "reshape",
        "transpose",
        "ravel",
        "flatten",
        "concatenate",
        "stack",
        "split",
        "squeeze",
        "dot",
        "matmul",
        "Series",
        "DataFrame",
    ],
)
print("Missing in cp:", missing_cp)

if hasattr(cp, "linalg"):
    missing_linalg = check_attrs(cp.linalg, ["inv", "det", "norm"])
    print("Missing in cp.linalg:", missing_linalg)
else:
    print("Missing cp.linalg completely")

if hasattr(cp, "random"):
    missing_random = check_attrs(cp.random, ["rand", "randn", "randint"])
    print("Missing in cp.random:", missing_random)
else:
    print("Missing cp.random completely")

arr = cp.array([1, 2, 3])
missing_ndarray = check_attrs(arr, ["shape", "ndim", "dtype", "size", "T", "astype"])
print("Missing in ndarray:", missing_ndarray)

df = cp.DataFrame({"a": [1, 2]})
missing_df = check_attrs(
    df,
    [
        "to_csv",
        "from_dict",
        "head",
        "tail",
        "loc",
        "iloc",
        "drop",
        "rename",
        "assign",
        "sort_values",
        "sort_index",
        "reset_index",
        "set_index",
        "astype",
        "copy",
        "apply",
        "sum",
        "mean",
        "std",
        "min",
        "max",
        "count",
        "value_counts",
        "describe",
        "groupby",
        "merge",
        "join",
        "pivot",
        "pivot_table",
    ],
)
print("Missing in DataFrame:", missing_df)
