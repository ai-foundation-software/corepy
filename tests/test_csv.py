import os
import tempfile

import pytest

import corepy as cp


def test_read_csv():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write("A,B\n1.0,10.0\n2.0,20.0\n3.0,30.0\n")
        tmp_path = f.name

    try:
        df = cp.read_csv(tmp_path)
        assert df.shape == (3, 2)
        assert set(df.columns) == {"A", "B"}
        assert df["A"].values.to_list() == [1.0, 2.0, 3.0]
        assert df["B"].values.to_list() == [10.0, 20.0, 30.0]
    finally:
        os.unlink(tmp_path)
