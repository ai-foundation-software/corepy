# ❓ Corepy FAQ & Troubleshooting Guide

**Version**: 0.3.2  
**Last Updated**: 2026-08-21  

---

## ❓ Frequently Asked Questions

### Q1: What is Corepy?
**A**: Corepy is a high-performance Python runtime backed by a pure Rust computational engine. It provides NumPy and Pandas compatible APIs (`ndarray`, `Series`, `DataFrame`, `UFuncs`) with automatic hardware acceleration (CPU SIMD, OpenBLAS, Metal GPU).

### Q2: What's new in v0.3.2?
**A**:
- **PyO3 Type Unwrapping**: Pure Python lists and lazy sequence inputs pass directly into PyO3 FFI routines (`cp.stack`, `cp.concatenate`).
- **2D Comparison Broadcasting**: Matrix comparisons (`arr2d > 2.0`) broadcast scalar values elementwise.
- **Boolean Mask Indexing**: Boolean masks slice ndarrays natively (`arr[arr > 2.0]`).
- **Series & DataFrame Fixes**: Implemented full statistical methods (`mean`, `std`, `var`, `sum`, `min`, `max`, `apply`, `map`, `filter`, `describe`).
- **Math Domain Guards**: Functions like `cp.arctanh(2.0)` or `cp.log(-1.0)` return `NaN` with standard `RuntimeWarning`s.
- **GPUBuffer Abstraction**: High-level GPU buffer manager with automatic CPU fallback.

### Q3: Why do I see a `RuntimeWarning: invalid value encountered in <op>`?
**A**: This warning occurs when input values fall outside the mathematical domain of an operation (e.g. `cp.arctanh(2.0)` or `cp.log(-1.0)`). Corepy follows NumPy semantics, emitting a warning and returning `float("nan")`.

---

## 🛠️ Troubleshooting Common Issues

### Issue 1: `TypeError: argument 'arrays': 'None' is not an instance of '_RustCoreArray'`
- **Cause**: Passing pure Python lists or uninitialized arrays to `cp.stack()`.
- **Solution**: Upgrade to Corepy v0.3.2, which automatically calls `_ensure_core_array()` during PyO3 invocation.

### Issue 2: `OverflowError: can't convert negative int to unsigned`
- **Cause**: Passing negative bounds to `Series.tail(n)` or `Series.head(n)` (e.g., `s.tail(-2)`).
- **Solution**: Upgraded in v0.3.2 using signed `isize` PyO3 conversions.

### Issue 3: `ValueError: Cannot compare non-scalar array`
- **Cause**: Comparing 2D or multi-dimensional arrays against scalar values (e.g. `arr2d > 2`).
- **Solution**: Fixed in v0.3.2 via Rust `elementwise_cmp` scalar broadcasting.

---

## 📞 Support & Community
- **GitHub Issues**: Report bugs or feature requests on our repository.
- **Contributing**: See [CONTRIBUTING.md](../07_contributing/CONTRIBUTING.md) for contribution guidelines.
