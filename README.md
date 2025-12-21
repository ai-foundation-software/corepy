# corepy

A unified, high-performance core runtime for data, computation, and AI workflows.

## Usage

```bash
# Install the package
uv pip install .

# Run the example
uv run examples/hello_corepy.py
```

```python
import corepy

# Use the optimized C++ kernel
result = corepy.add_one(10)
print(result) # 11
```

