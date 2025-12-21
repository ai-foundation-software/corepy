from corepy.data import Table
from corepy.schema import Schema

def test_table_creation():
    data = [{"a": 1}, {"a": 2}]
    t = Table(data)
    assert len(t) == 2
    assert t.to_list() == data
    assert t.schema is None

def test_table_repr():
    data = [{"a": 1}]
    t = Table(data)
    assert "Table(rows=1" in repr(t)
    assert "schema=None" in repr(t)
from corepy.schema import Schema

def test_table_creation(sample_data):
    table = Table(sample_data)
    assert len(table) == 2
    assert table.to_list() == sample_data

def test_table_with_schema():
    schema = Schema(fields=[])
    schema.add_field("id", "int")
    data = [{"id": 1}]
    table = Table(data, schema=schema)
    assert table.schema is not None
