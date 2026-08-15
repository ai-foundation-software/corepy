# Examples

Practical examples demonstrating Corepy features.

## Available Examples

### Comprehensive Demonstrations

- **`cpu_features_demo.py`** - Complete demonstration of all CPU-optimized features including:
  - Element-wise operations (add, sub, mul, div) with AVX2 SIMD
  - Reduction operations (sum, mean) with SIMD optimization
  - Matrix operations (dot product, matmul)
  - Backend system and profiling
  - Performance scaling analysis

### Validated Examples

- **`working_examples.py`** - Basic array operations and usage patterns
- **`validated_examples.py`** - Tested use cases with expected outputs
- **`new_features_demo.py`** - Demonstrates `concatenate` and `compute_stats` (New in 0.3.0)

## Running Examples

```bash
# Run comprehensive CPU demo
python examples/cpu_features_demo.py

# Run basic examples
python examples/working_examples.py

# Run validated examples
python examples/validated_examples.py

# Run new features demo
python examples/new_features_demo.py
```

## Example Output

Each example includes:
- Feature demonstration with code
- Expected vs actual results
- Performance measurements
- Practical use cases

## Contributing Examples

When adding new examples:
1. Keep examples focused on a single concept
2. Include clear comments explaining each step
3. Show expected output
4. Test thoroughly before committing
