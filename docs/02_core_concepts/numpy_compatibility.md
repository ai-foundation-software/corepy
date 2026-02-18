# NumPy Compatibility Guide

CorePy aims to provide a familiar API for NumPy users. This table tracks the parity between NumPy and CorePy APIs.

| NumPy API | CorePy API | Status | Notes |
|:----------|:-----------|:-------|:------|
| **Creation** | | | |
| `np.array()` | `cp.array()` | ✅ Supported | Primary factory function |
| `np.zeros()` | `cp.zeros()` | ✅ Supported | |
| `np.ones()` | `cp.ones()` | ✅ Supported | |
| `np.empty()` | `cp.empty()` | ✅ Supported | |
| `np.arange()` | `cp.arange()` | ✅ Supported | |
| `np.linspace()` | - | 🚧 Planned | |
| `np.eye()` | - | 🚧 Planned | |
| **Array Properties** | | | |
| `.shape` | `.shape` | ✅ Supported | |
| `.dtype` | `.dtype` | ✅ Supported | Returns `cp.DataType` enum |
| `.ndim` | `.ndim` | ✅ Supported | |
| `.size` | `.size` | ✅ Supported | |
| `.T` | `.T` | ✅ Supported | 2D arrays only currently |
| **Manipulation** | | | |
| `.reshape()` | `.reshape()` | ✅ Supported | |
| `.transpose()` | `.transpose()` | ✅ Supported | |
| `.flatten()` | - | 🚧 Planned | Use `.reshape(-1)` equivalent |
| `np.concatenate()` | - | 🚧 Planned | |
| `np.stack()` | - | 🚧 Planned | |
| **Math** | | | |
| `np.add()` | `cp.add()` | ✅ Supported | Also `+` operator |
| `np.subtract()` | - | 🚧 Planned | Use `-` operator |
| `np.multiply()` | - | 🚧 Planned | Use `*` operator |
| `np.divide()` | - | 🚧 Planned | Use `/` operator |
| `np.matmul()` | `cp.matmul()` | ✅ Supported | Also `@` operator |
| `np.dot()` | `cp.dot()` | ✅ Supported | Alias for matmul |
| `np.sum()` | `arr.sum()` | ✅ Supported | Method only currently |
| `np.mean()` | `arr.mean()` | ✅ Supported | Method only currently |
| `np.std()` | `arr.std()` | ✅ Supported | Method only currently |
| `np.min()` | `arr.min()` | ✅ Supported | Method only currently |
| `np.max()` | `arr.max()` | ✅ Supported | Method only currently |
| **Indexing** | | | |
| Basic Indexing | - | 🚧 Planned | `arr[0]` not yet fully supported |
| Slicing | - | 🚧 Planned | `arr[0:5]` not yet fully supported |
| Boolean Indexing | - | 🚧 Planned | |

## Key Differences

1.  **Backend Selection**: CorePy arrays automatically select the best backend (CPU/Metal) based on size and operation.
2.  **Data Types**: CorePy uses `cp.DataType` enum (e.g., `cp.float32`) instead of numpy types, though numpy types are accepted in factory functions.
3.  **In-Place Ops**: CorePy operations currently return new arrays (functional style). In-place mutation is planned.
