from corepy.data import Table
from corepy.runtime.pipeline import Pipeline

def test_pipeline_creation():
    p = Pipeline()
    assert len(p.steps) == 0

def test_pipeline_add_step():
    p = Pipeline()
    p.add_step(lambda x: x)
    assert len(p.steps) == 1

def test_pipeline_repr():
    p = Pipeline()
    assert "Pipeline(steps=0)" in repr(p)

def test_pipeline_empty_pipeline_execution():
    data = [{"val": 1}]
    t = Table(data)
        
    p = Pipeline(steps=[])
    res = p.run(t)
    assert res is t  # An empty pipeline should return the input table

def test_pipeline_execution(sample_data):
    def add_processed_flag(table: Table) -> Table:
        data = table.to_list()
        for row in data:
            row["processed"] = True
        return Table(data, schema=table.schema)

    pipeline = Pipeline()
    pipeline.add_step(add_processed_flag)
    
    input_table = Table(sample_data)
    result = pipeline.run(input_table)
    
    for row in result.to_list():
        assert row["processed"] is True
