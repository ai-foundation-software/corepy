import pytest

@pytest.fixture
def sample_data():
    return [
        {"id": 1, "name": "Alice", "score": 90.5},
        {"id": 2, "name": "Bob", "score": 85.0},
    ]
