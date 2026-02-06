# FAQ & Troubleshooting

## General

### Why "Corepy"?
We wanted a core runtime that was simpler than PyTorch but faster/safer than raw NumPy for specific deployment scenarios.

### Is this a replacement for NumPy/Pandas?
No.
- Use **Pandas** for messy data cleaning, CSV parsing, and exploration.
- Use **NumPy** for general scientific computing.
- Use **Corepy** when you need a production-grade tensor runtime with strict typing, predictable memory usage, and built-in profiling.

## Technical

### Why do I see `add_one`?
That is a legacy test function from the initial skeleton. It will be removed in v1.0.

### Why is `read_csv` missing?
In v0.2.2, we removed the experimental Python-based CSV reader to focus on the Tensor core. We recommend using `pandas.read_csv()` to load data, then converting to `corepy.Tensor(df.values)` for processing.

### Why strict Float32?
Most AI and signal processing workloads do not need Double Precision (`Float64`), which consumes 2x memory and bandwidth. Defaults matter. You can explicitly use `DataType.FLOAT64` if needed.

## Troubleshooting

### `ImportError: C++ extension not loaded`
This usually means the wheel was built without compiling the C++ component or the shared library cannot be found.
- **Fix**: Reinstall with verbose output: `pip install -v .`
- **Dev**: Ensure `cmake` is in your PATH.

### Performance seems slow?
- Check if you are running on a very small tensor. The overhead of crossing Python->Rust->C++ dominates for arrays smaller than ~1000 elements.
- Use `cp.enable_profiling()` to see if `Python Fallback` is being triggered (which happens if the Rust extension fails to load).
