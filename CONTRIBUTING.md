# 👋 Contributing to Corepy

First off, thank you for considering contributing to Corepy! Open source lives because of people like you.

Whether you are fixing a typo, adding an example, or writing Rust kernels, we welcome your help.

---

## 🌟 Our Philosophy

1.  **Correctness First**: It is better to be slow and right than fast and wrong.
2.  **Rust First**: New compute kernels should be written in Rust. We are actively migrating legacy C/C++ code to Rust.
3.  **Device Aware**: Code should gracefully handle different backends (CPU, GPU, Metal).
4.  **Test Everything**: If it's not tested, it's broken.

---

## 🦀 Why Rust?

We are transitioning from C/C++ to Rust for our compute kernels because:

- **Memory Safety**: No segfaults, no data races — guaranteed at compile time.
- **Performance**: Zero-cost abstractions with SIMD intrinsics support.
- **Modern Tooling**: `cargo`, `clippy`, and `maturin` provide excellent developer experience.
- **PyO3 Integration**: Seamless Python bindings via the `pyo3` crate.

> **Note**: The `csrc/` directory contains legacy C++ kernels. New contributions should target `rust/core/` instead.

---

## 🛠️ Developer Setup

For detailed setup instructions for Windows, Linux, and macOS, please see **[DEVELOPMENT.md](DEVELOPMENT.md)**.

**Quick Layout**:
- **Python**: `corepy/`
- **Rust**: `rust/`
- **C++**: `csrc/`

**Quick Start**:
```bash
make install
make test
```

---

## 📝 How to Submit a Change

1.  **Find an Issue**: Look for issues labeled `good first issue` or `rust-migration` on GitHub.
2.  **Create a Branch**: `git checkout -b my-new-feature`
3.  **Make your Changes**:
    - Python code: `ruff check . && ruff format .`
    - Rust code: `cargo fmt && cargo clippy`
4.  **Add Tests**:
    - Rust unit tests in `rust/core/src/`
    - Integration tests in `tests/`
5.  **Submit a PR**: Push your branch and open a Pull Request.

---

## 📂 Project Structure

| Directory | Language | Description |
|-----------|----------|-------------|
| `corepy/` | Python | User-facing API |
| `rust/core/` | Rust | **Primary runtime** — kernels, memory, dispatch |
| `csrc/` | C++ | ⚠️ Legacy kernels (being migrated to Rust) |
| `tests/` | Python | Test suite |
| `docs/` | Markdown | Documentation |

---

## 🚀 High-Impact Contributions

We especially welcome help with:

- **Rust kernel ports**: Migrate operations from `csrc/` to `rust/core/src/`
- **SIMD optimizations**: Leverage `std::arch` for AVX2/NEON intrinsics in Rust
- **GPU backends**: Metal/CUDA integration via Rust FFI

---

## 🆘 Getting Help

If you get stuck, please open an issue or ask in our GitHub Discussions.
