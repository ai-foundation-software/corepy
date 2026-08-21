from typing import Any, List, Optional, Union

from ._corepy_rust import _RustCoreArray as CoreArray
from ._corepy_rust import _RustSeries
from .array import ndarray
from .backend.types import DataType


class Series:
    """
    Pandas-compatible Series implementation backed by CorePy Rust engine.
    """

    def __init__(
        self,
        data: Union[List[Any], ndarray, "Series", _RustSeries],
        index: Optional[List[str]] = None,
        name: str = "",
    ):
        if isinstance(data, _RustSeries):
            self._series = data
            return

        # Prepare name
        name_str = str(name)

        # Prepare data as CoreArray
        if isinstance(data, ndarray):
            if data._core_array is not None:
                core_data = data._core_array
            else:
                flat_data = [float(x) for x in data.to_list()]
                core_data = CoreArray(flat_data, [len(flat_data)])
        elif isinstance(data, Series):
            core_data = data._series.values
            if index is None:
                index = data.index
            name_str = data.name if not name else name_str
        elif isinstance(data, list):
            # Simple inference for now
            is_str = any(isinstance(x, str) for x in data)
            is_float = any(isinstance(x, float) for x in data) if not is_str else False

            # Use CoreArray from rust backend
            if is_str:
                core_data = CoreArray.from_strings([str(x) for x in data], [len(data)])
            elif is_float:
                core_data = CoreArray([float(x) for x in data], [len(data)])
            else:
                # Temporary workaround: push to float for now until int support is fully fleshed out in py wrapper for CoreArray
                core_data = CoreArray([float(x) for x in data], [len(data)])
        else:
            raise TypeError(f"Unsupported data type for Series: {type(data)}")

        # Construct Rust Series
        if index is None:
            self._series = _RustSeries(name_str, core_data)
        else:
            self._series = _RustSeries(name_str, core_data, index)

    @property
    def name(self) -> str:
        return self._series.name

    @name.setter
    def name(self, value: str):
        self._series.name = str(value)

    @property
    def index(self) -> List[str]:
        return self._series.index

    @property
    def values(self) -> ndarray:
        """Returns the underlying data as a corepy.ndarray"""
        ct = self._series.data

        # Check dtype string returned by series.dtype
        dt_str = self._series.dtype.lower()
        if "string" in dt_str:
            dt = DataType.STRING
        elif "float" in dt_str:
            dt = DataType.FLOAT32
        else:
            dt = DataType.INT32

        return ndarray._wrap_core_array(ct, dtype=dt)

    @property
    def dtype(self) -> str:
        # the rust series dtype() returns a string like "Float32" or "Int32"
        return self._series.dtype

    def head(self, n: int = 5) -> "Series":
        return Series(self._series.head(n))

    def tail(self, n: int = 5) -> "Series":
        return Series(self._series.tail(n))

    def unique(self) -> "Series":
        return Series(self._series.unique())

    def value_counts(self) -> dict:
        return self._series.value_counts()

    def _scalar(self, val: Any) -> Any:
        if hasattr(val, "to_list"):
            l = val.to_list()
            while isinstance(l, list) and len(l) > 0:
                l = l[0]
            return l
        return val

    def mean(self) -> float:
        return float(self._scalar(self.values.mean()))

    def std(self, ddof: int = 1) -> float:
        return float(self._scalar(self.values.std(ddof=ddof)))

    def var(self, ddof: int = 1) -> float:
        return float(self._scalar(self.values.var(ddof=ddof)))

    def sum(self) -> float:
        return float(self._scalar(self.values.sum()))

    def min(self) -> Any:
        return self._scalar(self.values.min())

    def max(self) -> Any:
        return self._scalar(self.values.max())

    def tolist(self) -> list:
        return self.values.to_list()

    def apply(self, func: Any) -> "Series":
        vals = [func(x) for x in self.values.to_list()]
        return Series(vals, index=self.index, name=self.name)

    def map(self, arg: Any) -> "Series":
        if isinstance(arg, dict):
            vals = [arg.get(x, x) for x in self.values.to_list()]
        elif callable(arg):
            vals = [arg(x) for x in self.values.to_list()]
        else:
            raise TypeError("map argument must be a dict or callable")
        return Series(vals, index=self.index, name=self.name)

    def filter(
        self, items=None, like: Optional[str] = None, regex: Optional[str] = None
    ) -> "Series":
        indices = self.index
        vals = self.values.to_list()
        filtered_indices = []
        filtered_vals = []
        import re

        for idx, val in zip(indices, vals):
            keep = False
            if items is not None and idx in items:
                keep = True
            elif like is not None and like in str(idx):
                keep = True
            elif regex is not None and re.search(regex, str(idx)):
                keep = True
            if keep:
                filtered_indices.append(idx)
                filtered_vals.append(val)
        return Series(filtered_vals, index=filtered_indices, name=self.name)

    def __len__(self) -> int:
        return len(self._series.index)

    def __repr__(self) -> str:
        # Simple representation inspired by pandas
        max_rows = 10
        total_rows = len(self)

        if total_rows == 0:
            return f"Series([], Name: {self.name}, dtype: {self.dtype})"

        lines = []
        if total_rows <= max_rows:
            # Show all
            vals = self.values.to_list()
            for idx, val in zip(self.index, vals):
                lines.append(f"{idx}\t{val}")
        else:
            # Show head and tail
            head = self.head(5)
            tail = self.tail(5)

            head_vals = head.values.to_list()
            for idx, val in zip(head.index, head_vals):
                lines.append(f"{idx}\t{val}")

            lines.append("...")

            tail_vals = tail.values.to_list()
            for idx, val in zip(tail.index, tail_vals):
                lines.append(f"{idx}\t{val}")

        res = "\n".join(lines)
        res += f"\nName: {self.name}, dtype: {self.dtype}"
        return res

    def __getstate__(self):
        return {"_series": self._series}

    def __setstate__(self, state):
        self._series = state["_series"]
