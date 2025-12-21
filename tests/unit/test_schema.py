from corepy.schema import Schema, Field

def test_schema_creation():
    s = Schema(fields=[])
    assert s.fields == []

def test_schema_add_field():
    s = Schema(fields=[])
    s.add_field("age", "int")
    assert len(s.fields) == 1
    assert s.fields[0].name == "age"
    assert s.fields[0].dtype == "int"

def test_schema_get_field():
    s = Schema(fields=[])
    s.add_field("age", "int")
    
    f = s.get_field("age")
    assert f is not None
    assert f.name == "age"
    
    f_missing = s.get_field("invalid")
    assert f_missing is None

def test_schema_repr():
    s = Schema(fields=[])
    assert "Schema(fields=[])" in repr(s)

def test_schema_definition():
    schema = Schema(fields=[])
    schema.add_field("id", "int")
    schema.add_field("name", "str")
    
    assert len(schema.fields) == 2
    field = schema.get_field("id")
    assert field.name == "id"
    assert field.dtype == "int"

def test_schema_field_metadata():
    schema = Schema(fields=[])
    schema.add_field("score", "float", metadata={"unit": "points"})
    field = schema.get_field("score")
    assert field.metadata["unit"] == "points"
