import pytest

import corepy as cp


def test_groupby_sum():
    df = cp.DataFrame(
        {"Key": [1.0, 2.0, 1.0, 2.0, 3.0], "Value": [10.0, 20.0, 30.0, 40.0, 50.0]}
    )

    gb_sum = df.groupby("Key").sum()

    assert gb_sum.shape == (3, 2)
    assert set(gb_sum.columns) == {"Key", "Value"}

    # Keys should be sorted based on our implementation
    assert gb_sum["Key"].values.to_list() == [1.0, 2.0, 3.0]
    assert gb_sum["Value"].values.to_list() == [40.0, 60.0, 50.0]


def test_groupby_mean():
    df = cp.DataFrame(
        {"Key": [1.0, 1.0, 2.0, 2.0, 2.0], "Value": [10.0, 30.0, 30.0, 10.0, 20.0]}
    )

    gb_mean = df.groupby("Key").mean()

    # Keys should be sorted based on our implementation
    assert gb_mean["Key"].values.to_list() == [1.0, 2.0]
    assert gb_mean["Value"].values.to_list() == [20.0, 20.0]
