import math

import pytest

import corepy as cp


def test_pivot():
    df = cp.DataFrame(
        {
            "Date": [1.0, 1.0, 2.0, 2.0],
            "City": [10.0, 20.0, 10.0, 20.0],
            "Temp": [32.0, 45.0, 35.0, 42.0],
        }
    )

    pivoted = df.pivot(index="Date", columns="City", values="Temp")

    assert pivoted.shape == (2, 3)
    assert set(pivoted.columns) == {"Date", "10", "20"}

    assert pivoted["Date"].values.to_list() == [1.0, 2.0]
    assert pivoted["10"].values.to_list() == [32.0, 35.0]
    assert pivoted["20"].values.to_list() == [45.0, 42.0]


def test_pivot_missing():
    df = cp.DataFrame({"Date": [1.0, 2.0], "City": [10.0, 20.0], "Temp": [32.0, 45.0]})

    pivoted = df.pivot(index="Date", columns="City", values="Temp")

    assert pivoted.shape == (2, 3)
    assert set(pivoted.columns) == {"Date", "10", "20"}

    assert pivoted["Date"].values.to_list() == [1.0, 2.0]

    # Missing values should be NaN
    col10 = pivoted["10"].values.to_list()
    assert col10[0] == 32.0
    assert math.isnan(col10[1])

    col20 = pivoted["20"].values.to_list()
    assert math.isnan(col20[0])
    assert col20[1] == 45.0
