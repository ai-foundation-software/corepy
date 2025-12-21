from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field as PydanticField

class Field(BaseModel):
    """
    Represents a single field in a schema.
    """
    name: str
    dtype: str
    metadata: Dict[str, Any] = PydanticField(default_factory=dict)

class Schema(BaseModel):
    """
    Defines the structure of a dataset.
    """
    fields: List[Field]

    def add_field(self, name: str, dtype: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Adds a field to the schema.
        """
        self.fields.append(Field(name=name, dtype=dtype, metadata=metadata or {}))

    def get_field(self, name: str) -> Optional[Field]:
        """
        Retrieves a field by name.
        """
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def __repr__(self) -> str:
        return f"Schema(fields={self.fields})"
