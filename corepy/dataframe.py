from typing import Any, Dict, List, Optional, Tuple, Union

from ._corepy_rust import _RustDataFrame, _RustGroupBy
from ._corepy_rust import read_csv as _rust_read_csv
from .array import ndarray
from .series import Series


def read_csv(filepath: str) -> "DataFrame":
    rust_df = _rust_read_csv(filepath)
    return DataFrame(rust_df)


class GroupBy:
    """
    Pandas-compatible GroupBy, backed by Rust Hash Engine.
    """

    def __init__(self, rust_groupby: _RustGroupBy):
        self._groupby = rust_groupby

    def sum(self) -> "DataFrame":
        return DataFrame(self._groupby.sum())

    def mean(self) -> "DataFrame":
        return DataFrame(self._groupby.mean())

    def agg(self, agg_dict: Dict[str, str]) -> "DataFrame":
        # Simplified agg matching only single operations for the tutorial
        # since Rust backend handles mean/sum globally for the group
        ops = list(agg_dict.values())
        if "mean" in ops:
            return self.mean()
        elif "sum" in ops:
            return self.sum()
        else:
            raise NotImplementedError(f"Unsupported aggregations: {ops}")


class DataFrame:
    """
    Pandas-compatible DataFrame backed by CorePy Rust Engine.
    """

    def __init__(self, data=None):
        self._df = _RustDataFrame()
        if data is not None:
            if isinstance(data, dict):
                for col_name, col_data in data.items():
                    if isinstance(col_data, Series):
                        self._df.insert(str(col_name), col_data._series)
                    else:
                        s = Series(col_data, name=str(col_name))
                        self._df.insert(str(col_name), s._series)
            elif isinstance(data, _RustDataFrame):
                self._df = data
            elif isinstance(data, DataFrame):
                for col_name in data.columns:
                    s = data[col_name]
                    self._df.insert(col_name, s._series)

    @classmethod
    def from_dict(cls, data: dict) -> "DataFrame":
        return cls(data)

    def to_csv(self, filepath: str):
        # MVP: pure python CSV write
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
            # Assuming row-wise fetching is slow but works
            for i in range(self.shape[0]):
                row = [self[col].values.to_list()[i] for col in self.columns]
                writer.writerow(row)

    def head(self, n: int = 5) -> "DataFrame":
        return self.iloc[:n]

    def tail(self, n: int = 5) -> "DataFrame":
        return self.iloc[-n:]

    @property
    def loc(self):
        class _LocIndexer:
            def __init__(self, df):
                self.df = df

            def __getitem__(self, key):
                if isinstance(key, slice):
                    # Mocking loc slice as iloc for the dummy API since we don't have labeled indices
                    return self.df.iloc[key]
                return self.df.iloc[key : key + 1]

        return _LocIndexer(self)

    def assign(self, **kwargs) -> "DataFrame":
        # Copy data
        new_df = DataFrame()
        for col in self.columns:
            new_df[col] = self[col]
        for k, v in kwargs.items():
            new_df[k] = v
        return new_df

    def sort_index(self) -> "DataFrame":
        return self  # No actual index

    def reset_index(self, drop: bool = False) -> "DataFrame":
        return self  # No actual index

    def set_index(self, keys) -> "DataFrame":
        return self

    def astype(self, dtype) -> "DataFrame":
        new_df = DataFrame()
        for col in self.columns:
            new_df[col] = self[col].astype(dtype)
        return new_df

    def copy(self) -> "DataFrame":
        new_df = DataFrame()
        for col in self.columns:
            new_df[col] = self[col]
        return new_df

    def apply(self, func, axis=0) -> "DataFrame":
        raise NotImplementedError("Apply is stubbed")

    def sum(self) -> "Series":
        data = {col: self[col].sum() for col in self.columns}
        return Series(list(data.values()))

    def mean(self) -> "Series":
        data = {col: self[col].mean() for col in self.columns}
        return Series(list(data.values()))

    def std(self) -> "Series":
        data = {col: self[col].std() for col in self.columns}
        return Series(list(data.values()))

    def min(self) -> "Series":
        data = {col: self[col].min() for col in self.columns}
        return Series(list(data.values()))

    def max(self) -> "Series":
        data = {col: self[col].max() for col in self.columns}
        return Series(list(data.values()))

    def count(self) -> "Series":
        data = {col: self.shape[0] for col in self.columns}
        return Series(list(data.values()))

    def value_counts(self) -> "Series":
        raise NotImplementedError("value_counts is stubbed")

    def describe(self) -> "DataFrame":
        """Generate descriptive statistics."""
        data = {}
        for col in self.columns:
            s = self[col]
            data[col] = Series(
                [self.shape[0], s.mean(), s.std(), s.min(), s.max()]
            )  # Simplified
        return DataFrame(data)

    def join(self, other: "DataFrame", on: str, how: str = "left") -> "DataFrame":
        return self.merge(other, left_on=on, right_on=on, how=how)

    def pivot_table(
        self, values=None, index=None, columns=None, aggfunc="mean"
    ) -> "DataFrame":
        return self.pivot(index, columns, values)

    @property
    def columns(self) -> List[str]:
        return list(self._df.columns.keys())

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._df.row_count, len(self._df.columns))

    def __getitem__(self, key: Union[str, List[str]]):
        if isinstance(key, str):
            rust_series = self._df.get_column(key)
            return Series(rust_series)
        elif isinstance(key, list):
            new_df = DataFrame()
            for col in key:
                if not isinstance(col, str):
                    raise TypeError("Column name must be string")
                rust_series = self._df.get_column(col)
                new_df._df.insert(col, rust_series)
            return new_df
        else:
            raise TypeError("Index must be string or list of strings")

    def __setitem__(self, key: str, value):
        if not isinstance(key, str):
            raise TypeError("Column name must be a string")
        if isinstance(value, Series):
            self._df.insert(key, value._series)
        else:
            s = Series(value, name=key)
            self._df.insert(key, s._series)

    def add_int_column(self, name: str, data: list):
        self[name] = data

    def add_float_column(self, name: str, data: list):
        self[name] = data

    def drop(self, columns: Union[str, List[str]]) -> "DataFrame":
        cols = [columns] if isinstance(columns, str) else columns
        new_rust_df = _RustDataFrame()
        for col_name, series in self._df.columns.items():
            if col_name not in cols:
                new_rust_df.insert(col_name, series)
        return DataFrame(new_rust_df)

    def rename(self, columns: Dict[str, str]) -> "DataFrame":
        new_rust_df = _RustDataFrame()
        for col_name, series in self._df.columns.items():
            if col_name in columns:
                # pass cloned rust series, note that the cloned series name must also be updated
                # doing this by creating a fresh Series in Python is easiest to update name
                s = Series(series)
                s.name = columns[col_name]
                new_rust_df.insert(columns[col_name], s._series)
            else:
                new_rust_df.insert(col_name, series)
        return DataFrame(new_rust_df)

    def sort_values(self, by: str, ascending: bool = True) -> "DataFrame":
        return DataFrame(self._df.sort_values(by, ascending))

    def filter(self, col_name: str, value: Any, op: str = "==") -> "DataFrame":
        s = self[col_name]
        if "string" in s.dtype.lower():
            # filter_eq in Rust handles strings too if we pass them correctly via PyO3
            return DataFrame(self._df.filter_eq(col_name, str(value)))
        return DataFrame(self._df.filter_eq(col_name, float(value)))

    def groupby(self, by: str) -> GroupBy:
        return GroupBy(self._df.groupby(by))

    def merge(
        self, right: "DataFrame", left_on: str, right_on: str, how: str = "inner"
    ) -> "DataFrame":
        return DataFrame(self._df.join(right._df, left_on, right_on, how))

    def filter_int_eq(self, col_name: str, value: int) -> "DataFrame":
        return self.filter(col_name, value)

    def pivot(self, index: str, columns: str, values: str) -> "DataFrame":
        return DataFrame(self._df.pivot(index, columns, values))

    @property
    def iloc(self):
        class _ILocIndexer:
            def __init__(self, df):
                self.df = df

            def __getitem__(self, key):
                if isinstance(key, slice):
                    s = key.start
                    e = key.stop
                    # Convert negative slices
                    n = self.df.shape[0]
                    if s is not None and s < 0:
                        s += n
                    if e is not None and e < 0:
                        e += n
                    new_rust_df = self.df._df.iloc(s, e)
                    return DataFrame(new_rust_df)
                elif isinstance(key, int):
                    # Single row slice
                    i = key
                    n = self.df.shape[0]
                    if i < 0:
                        i += n
                    new_rust_df = self.df._df.iloc(i, i + 1)
                    return DataFrame(new_rust_df)
                raise TypeError(
                    "iloc currently only supports slice objects or integers"
                )

        return _ILocIndexer(self)

    def __repr__(self) -> str:
        s = f"DataFrame({self.shape[0]} rows x {self.shape[1]} columns)\n"
        s += "Columns: " + ", ".join(self.columns)
        return s

    def __getstate__(self):
        return {"_df": self._df}

    def __setstate__(self, state):
        self._df = state["_df"]
