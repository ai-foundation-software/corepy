import corepy as cp

def test_end_to_end_flow():
    # 1. Define Schema
    schema = cp.schema.Schema(fields=[])
    schema.add_field("id", "int")
    schema.add_field("value", "float")

    # 2. Create Data
    data = [{"id": i, "value": float(i)} for i in range(5)]
    table = cp.data.Table(data, schema=schema)

    # 3. Process
    pipeline = cp.runtime.Pipeline()
    pipeline.run(table)
    
    assert len(table) == 5
