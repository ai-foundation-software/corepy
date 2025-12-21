from typing import Any, Dict, List, Optional
from corepy.schema import Schema

class Table:
    """
    A unified data container for tabular data.
    """
    def __init__(self, data: List[Dict[str, Any]], schema: Optional[Schema] = None):
        self._data = data
        self._schema = schema
        self._validate()

    def _validate(self) -> None:
        """
        Validates data against the schema if provided.
        """
        if self._schema:
            # perform basic validation
            pass

    @property
    def schema(self) -> Optional[Schema]:
        """
        Returns the table schema.
        """
        return self._schema

    def to_list(self) -> List[Dict[str, Any]]:
        """
        Returns the data as a list of dictionaries.
        """
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Table(rows={len(self)}, schema={self._schema})"
