import pytest

import corepy as cp


def test_series_creation_from_list():
    s = cp.Series([1.0, 2.0, 3.0], name="test")
    assert s.name == "test"
    assert len(s) == 3
    assert s.dtype == "Float32"
    assert s.values.to_list() == [1.0, 2.0, 3.0]
    assert s.index == ["0", "1", "2"]


def test_series_creation_from_ndarray():
    arr = cp.array([4.0, 5.0, 6.0])
    s = cp.Series(arr, index=["a", "b", "c"], name="from_arr")
    assert s.name == "from_arr"
    assert len(s) == 3
    assert s.values.to_list() == [4.0, 5.0, 6.0]
    assert s.index == ["a", "b", "c"]


def test_series_head_tail():
    s = cp.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    h = s.head(3)
    assert h.values.to_list() == [1.0, 2.0, 3.0]
    assert len(h) == 3

    t = s.tail(2)
    assert t.values.to_list() == [6.0, 7.0]
    assert len(t) == 2


def test_series_unique_value_counts():
    s = cp.Series([1.0, 2.0, 2.0, 3.0, 1.0, 1.0])
    u = s.unique()
    assert sorted(u.values.to_list()) == [1.0, 2.0, 3.0]

    vc = s.value_counts()
    assert vc[1.0] == 3
    assert vc[2.0] == 2
    assert vc[3.0] == 1
