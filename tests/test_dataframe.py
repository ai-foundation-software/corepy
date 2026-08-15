import pytest

import corepy as cp


def test_dataframe_creation():
    df = cp.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})

    assert df.shape == (3, 2)
    assert set(df.columns) == {"A", "B"}
    assert df["A"].values.to_list() == [1.0, 2.0, 3.0]


def test_dataframe_drop_rename():
    df = cp.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0], "C": [5.0, 6.0]})

    df2 = df.drop("B")
    assert set(df2.columns) == {"A", "C"}
    assert df2.shape == (2, 2)

    df3 = df.rename({"A": "X"})
    assert set(df3.columns) == {"X", "B", "C"}


def test_dataframe_iloc_filter_sort():
    df = cp.DataFrame({"A": [3.0, 1.0, 2.0], "B": [10.0, 20.0, 30.0]})

    sorted_df = df.sort_values("A")
    assert sorted_df["A"].values.to_list() == [1.0, 2.0, 3.0]
    assert sorted_df["B"].values.to_list() == [20.0, 30.0, 10.0]

    filtered_df = df.filter("A", 1.0)
    assert filtered_df.shape == (1, 2)
    assert filtered_df["B"].values.to_list() == [20.0]

    sliced_df = df.iloc[1:3]
    assert sliced_df.shape == (2, 2)
    assert sliced_df["A"].values.to_list() == [1.0, 2.0]
