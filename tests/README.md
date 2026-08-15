# Tests

Corepy test suite ensuring correctness and reliability.

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=corepy --cov-report=html

# Run specific test file
uv run pytest tests/test_array_integration.py

# Run specific test by name
uv run pytest -k "test_buffer_protocol"

# Verbose output
uv run pytest -v
```

## Test Organization

### Core Tests
- `test_array_integration.py` - Array integration tests
- `test_buffer_protocol.py` - Buffer protocol compliance
- `test_gap_analysis.py` - SIMD operation tests

### Unit Tests (`unit/`)
- `test_api.py` - Public API surface tests
- `test_data.py` - Data handling tests
- `test_pipeline.py` - Pipeline execution tests
- `test_schema.py` - Schema validation tests
- `test_init_fallback.py` - Initialization fallback tests

### Integration Tests (`integration/`)
- `test_flow.py` - End-to-end workflow tests
- `test_released_api.py` - Released API stability tests

### Backend Tests
- `test_backend_selector.py` - Backend selection logic
- `test_dispatch.py` - Operation dispatch tests
- `test_dispatch_matmul.py` - Matrix multiplication dispatch

### Performance Tests
- `bench_*.py` - Performance benchmarks
- `repro_strides.py` - Memory safety reproduction

## Test Coverage

Current: **58%**  
Target: **>80%** for core execution paths

Coverage by module:
- `array.py`: 65% (core logic)
- `backend/`: 60% (dispatch)
- `profiler/`: 21% (unused features)

## Writing Tests

1. Use `pytest` fixtures from `conftest.py`
2. Test both success and failure cases
3. Include edge cases (empty arrays, large arrays, etc.)
4. Use descriptive test names: `test_<feature>_<condition>_<expected>`
5. Add docstrings for complex tests

## CI/CD

Tests run automatically on:
- Pull requests
- Commits to main branch
- Release tags

All tests must pass before merging.
