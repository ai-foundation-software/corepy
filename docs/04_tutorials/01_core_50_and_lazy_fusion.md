# CorePy UFUNC CORE-50 & Lazy Fusion

This tutorial demonstrates the powerful **CORE-50 Operations** combined with **Lazy Execution Fusion**. 
By wrapping arrays in a `LazyArray`, CorePy can compile complex mathematical equations into a single Rust kernel pass, avoiding intermediate memory allocations.

## 1. Standard Eager Execution
Normally, CorePy executes eagerly like NumPy, which is already fast and SIMD-optimized.

```python
import math
import corepy as cp

# 10 Million elements
x = cp.array([float(i) for i in range(10_000_000)])

# This creates two temporary intermediate arrays!
eager_result = cp.sin(x) ** 2 + cp.cos(x) ** 2
```

## 2. Lazy Fusion

When you wrap the array in a `LazyArray`, operations aren't evaluated immediately. 
Instead, CorePy builds an execution graph. When `.compute()` is called, all operations are **topologically sorted and fused**.

```python
import corepy as cp
from corepy.lazy.array import LazyArray

# 10 Million elements
x = cp.array([float(i) for i in range(10_000_000)])

# Wrap in LazyArray
lx = LazyArray(x)

# Build the graph (Instantaneous)
lazy_equation = (lx.sin() ** 2) + (lx.cos() ** 2)

# Compile & Execute: Runs a single optimized Rust kernel without intermediate arrays!
result = lazy_equation.compute()

# Due to precision arithmetic, result is roughly ~ 1.0 for all elements
print(f"Mean result (should be 1.0): {result.mean()}")
```

### Supported CORE-50 Lazy Operations

You can seamlessly chain all of the new CORE-50 operations inside lazy evaluation:
*   Trigonometry (`sin`, `cos`, `tan`, `arcsin`...)
*   Exponential & Logarithmic (`exp`, `log`, `log2`...)
*   Rounding & Special (`floor`, `ceil`, `round_`, `square`, `reciprocal`)
*   Reductions (`sum`, `prod`, `mean`, `std`, `var`)
*   Arithmetic (`**`, `//`, `%`)
