import pytest

import corepy as cp


def test_merge_inner():
    df1 = cp.DataFrame({"ID": [1.0, 2.0, 3.0], "A": [10.0, 20.0, 30.0]})

    df2 = cp.DataFrame({"ID": [2.0, 3.0, 4.0], "B": [200.0, 300.0, 400.0]})

    merged = df1.merge(df2, left_on="ID", right_on="ID", how="inner")

    assert merged.shape == (2, 3)
    assert set(merged.columns) == {"ID", "A", "B"}
    assert merged["ID"].values.to_list() == [2.0, 3.0]
    assert merged["A"].values.to_list() == [20.0, 30.0]
    assert merged["B"].values.to_list() == [200.0, 300.0]


def test_merge_left():
    df1 = cp.DataFrame({"ID": [1.0, 2.0, 3.0], "A": [10.0, 20.0, 30.0]})

    df2 = cp.DataFrame({"ID": [2.0, 3.0, 4.0], "B": [200.0, 300.0, 400.0]})

    merged = df1.merge(df2, left_on="ID", right_on="ID", how="left")

    assert merged.shape == (3, 3)
    assert set(merged.columns) == {"ID", "A", "B"}
    assert merged["ID"].values.to_list() == [1.0, 2.0, 3.0]
    assert merged["A"].values.to_list() == [10.0, 20.0, 30.0]
    # For ID=1.0, B should be NaN
    import math

    assert math.isnan(merged["B"].values.to_list()[0])
    assert merged["B"].values.to_list()[1:] == [200.0, 300.0]
