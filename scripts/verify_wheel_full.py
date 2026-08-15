import os
import sys

import numpy as np

# 1. Environment and Isolation Check
try:
    import corepy as cp
except ImportError as e:
    print(f"CRITICAL: Failed to import corepy: {e}")
    sys.exit(1)

print(f"Loaded corepy from: {cp.__file__}")
if "site-packages" not in cp.__file__:
    print(
        "WARNING: corepy is NOT imported from clean virtual environment site-packages!"
    )
else:
    print("SUCCESS: Environment is isolated. corepy imported from site-packages.")

# 2. Public API Compliance Check
expected_cp_attrs = [
    "ndarray",
    "DataType",
    "array",
    "zeros",
    "ones",
    "full",
    "empty",
    "arange",
    "linspace",
    "eye",
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
    "where",
    "searchsorted",
    "take",
    "boolean_index",
    "sort",
    "argsort",
    "stable_sort",
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
    "enable_profiling",
    "disable_profiling",
    "detect_bottlenecks",
    "get_recommendations",
    "DataFrame",
    "Series",
    "read_csv",
    "dot",
    "matmul",
    "linalg",
    "get_backend_policy",
    "set_backend_policy",
    "get_device_info",
    "analyse_workload",
    "explain_last_dispatch",
]

missing_cp = []
for attr in expected_cp_attrs:
    if not hasattr(cp, attr):
        missing_cp.append(attr)

if missing_cp:
    print(f"FAILED: Missing attributes in corepy: {missing_cp}")
else:
    print("SUCCESS: All expected public APIs are present in the corepy root namespace.")

# Check linalg sub-namespace
expected_linalg_attrs = ["inv", "det", "norm"]
missing_linalg = []
if hasattr(cp, "linalg"):
    for attr in expected_linalg_attrs:
        if not hasattr(cp.linalg, attr):
            missing_linalg.append(attr)
    if missing_linalg:
        print(f"FAILED: Missing attributes in corepy.linalg: {missing_linalg}")
    else:
        print("SUCCESS: All expected corepy.linalg APIs are present.")
else:
    print("FAILED: corepy.linalg module is missing!")

# Check random sub-namespace
expected_random_attrs = ["rand", "randn", "randint"]
missing_random = []
if hasattr(cp, "random"):
    for attr in expected_random_attrs:
        if not hasattr(cp.random, attr):
            missing_random.append(attr)
    if missing_random:
        print(f"FAILED: Missing attributes in corepy.random: {missing_random}")
    else:
        print("SUCCESS: All expected corepy.random APIs are present.")
else:
    print("FAILED: corepy.random module is missing!")

# 3. Functional Testing and NumPy Compatibility Checks
failures = []


def run_test(name, fn):
    try:
        fn()
        print(f"  ✓ {name}: Passed")
    except Exception as e:
        print(f"  ✗ {name}: Failed: {e}")
        failures.append((name, str(e)))


# Test array creation and attributes (shape, ndim, size, dtype)
def test_creation_attributes():
    # Standard array creation
    arr = cp.array([1.0, 2.0, 3.0])
    np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    assert arr.shape == np_arr.shape, f"Shape mismatch: {arr.shape} vs {np_arr.shape}"
    assert arr.ndim == np_arr.ndim, f"Ndim mismatch: {arr.ndim} vs {np_arr.ndim}"
    assert arr.size == np_arr.size, f"Size mismatch: {arr.size} vs {np_arr.size}"

    # Factory functions
    z = cp.zeros((2, 3))
    assert z.shape == (2, 3)
    o = cp.ones((3, 2))
    assert o.shape == (3, 2)

    ar = cp.arange(10)
    assert ar.shape == (10,)
    assert ar.to_list() == list(range(10))


run_test("Creation and Basic Attributes", test_creation_attributes)


# Test mathematical operations & operator overloading
def test_arithmetic():
    a = cp.array([2.0, 4.0, 6.0])
    b = cp.array([1.0, 2.0, 3.0])

    # Operators
    assert (a + b).to_list() == [3.0, 6.0, 9.0]
    assert (a - b).to_list() == [1.0, 2.0, 3.0]
    assert (a * b).to_list() == [2.0, 8.0, 18.0]
    assert (a / b).to_list() == [2.0, 2.0, 2.0]

    # Functions
    assert cp.add(a, b).to_list() == [3.0, 6.0, 9.0]
    assert cp.subtract(a, b).to_list() == [1.0, 2.0, 3.0]
    assert cp.multiply(a, b).to_list() == [2.0, 8.0, 18.0]
    assert cp.divide(a, b).to_list() == [2.0, 2.0, 2.0]


run_test("Arithmetic Operations", test_arithmetic)


# Test broadcasting
def test_broadcasting():
    a = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    b = cp.array([10.0, 20.0, 30.0])  # (3,)

    # In NumPy, a + b adds [10, 20, 30] to each row of a
    res = a + b
    assert res.shape == (2, 3), f"Broadcasting shape mismatch: {res.shape}"
    assert res.to_list() == [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]], (
        f"Broadcasting values mismatch: {res.to_list()}"
    )


run_test("Broadcasting Support", test_broadcasting)


# Test linear algebra
def test_linalg():
    # Dot product
    v1 = cp.array([1.0, 2.0, 3.0])
    v2 = cp.array([4.0, 5.0, 6.0])
    res_dot = cp.dot(v1, v2)
    # The dot product of 1D vectors yields a scalar or 1-element array
    assert abs(res_dot.to_list()[0] - 32.0) < 1e-5

    # Matrix product
    m1 = cp.array([[1.0, 2.0], [3.0, 4.0]])
    m2 = cp.array([[5.0, 6.0], [7.0, 8.0]])
    res_at = m1 @ m2
    assert res_at.shape == (2, 2)
    assert res_at.to_list() == [[19.0, 22.0], [43.0, 50.0]]

    # Norm
    assert abs(cp.linalg.norm(v1) - np.linalg.norm([1.0, 2.0, 3.0])) < 1e-5

    # Det and Inv (if supported)
    # Let's check determinant
    det_val = cp.linalg.det(m1)
    assert abs(det_val - (-2.0)) < 1e-5

    # Inverse
    inv_m1 = cp.linalg.inv(m1)
    assert inv_m1.shape == (2, 2)
    # inv of [[1,2],[3,4]] is [[-2,1],[1.5,-0.5]]
    expected_inv = [[-2.0, 1.0], [1.5, -0.5]]
    for r1, r2 in zip(inv_m1.to_list(), expected_inv):
        for val1, val2 in zip(r1, r2):
            assert abs(val1 - val2) < 1e-5


run_test("Linear Algebra Operations", test_linalg)


# Test random generation
def test_random():
    r1 = cp.random.rand(2, 3)
    assert r1.shape == (2, 3)

    r2 = cp.random.randn(5)
    assert r2.shape == (5,)

    r3 = cp.random.randint(5, 10, (10,))
    assert r3.shape == (10,)
    assert all(5 <= x < 10 for x in r3.to_list())


run_test("Random Utilities", test_random)


# Test DataFrame / Series functionality
def test_dataframe_series():
    df = cp.DataFrame({"a": [1.0, 2.0, 1.0], "b": [10.0, 20.0, 30.0]})
    assert df.shape == (3, 2)
    assert "a" in df.columns
    assert "b" in df.columns

    # Selection
    col_a = df["a"]
    assert isinstance(col_a, cp.Series)
    assert len(col_a) == 3

    # Groupby
    gb = df.groupby("a")
    res = gb.sum()
    # a has values 1.0 (appears twice, sum = 40.0) and 2.0 (appears once, sum = 20.0)
    assert res.shape == (2, 2)


run_test("DataFrame/Series and GroupBy", test_dataframe_series)


# Test File I/O
def test_io():
    df = cp.DataFrame({"a": [1.1, 2.2], "b": [3.3, 4.4]})
    csv_file = "test_verify_io.csv"
    try:
        df.to_csv(csv_file)
        assert os.path.exists(csv_file)

        df_read = cp.read_csv(csv_file)
        assert df_read.shape == (2, 2)
        assert abs(df_read["a"].values.to_list()[0] - 1.1) < 1e-5
    finally:
        if os.path.exists(csv_file):
            os.remove(csv_file)


run_test("File I/O (CSV)", test_io)


# Test Lazy Evaluation
def test_lazy_eval():
    with cp.lazy():
        a = cp.array([10.0, 20.0, 30.0])
        b = a * 2.0
        c = b + 5.0
        res = c.compute()
    assert res.to_list() == [25.0, 45.0, 65.0]


run_test("Lazy Evaluation Mode", test_lazy_eval)


# Test Profiler
def test_profiler():
    cp.enable_profiling()
    a = cp.array([1.0, 2.0, 3.0])
    b = cp.array([4.0, 5.0, 6.0])
    c = a + b
    report = cp.profile_report()
    assert report is not None
    cp.disable_profiling()


run_test("Profiler API", test_profiler)

# Output summary
print("\n" + "=" * 40)
print(f"VERIFICATION SUMMARY: {len(failures)} failures out of 9 categories tested.")
if failures:
    print("FAILURES IDENTIFIED:")
    for name, err in failures:
        print(f"- {name}: {err}")
    sys.exit(1)
else:
    print(
        "ALL TESTS PASSED SUCCESSFULLY! The wheel complies with repository functionality."
    )
    sys.exit(0)
