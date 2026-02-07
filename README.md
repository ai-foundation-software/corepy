# Corepy
<h1 align="center">
<img src="assets/logo.svg" width="300" alt="Corepy Logo" style="background-color: white;">
</h1><br>

[![CI](https://github.com/ai-foundation-software/corepy/actions/workflows/ci.yml/badge.svg)](https://github.com/ai-foundation-software/corepy/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12%20|%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **High-Performance Tensor Runtime for Python.**
> *Rust-Powered Backend. Correctness First. Hardware Aware.*

## 📖 What is Corepy?

Corepy is a high-performance tensor library that bridges Python's ease of use with the raw speed and safety of **Rust** and **C++**.

Unlike untyped array libraries, Corepy is built on a **strict runtime** that ensures:
1.  **Safety**: Rust ownership guarantees prevent common segmentation faults.
2.  **Performance**: C++ kernels (AVX2, NEON) and **Metal** shaders for heavy number crunching.
3.  **Observability**: Zero-overhead built-in profiler to visualize bottlenecks.

It is designed for developers building **AI foundations**, **scientific simulations**, and **high-performance systems**.

### Key Features
- **NumPy-Compatible API**: Familiar `cp.array`, `cp.zeros`, `cp.add` interface.
- **🚀 Metal GPU**: Native acceleration on macOS (Apple Silicon) with `device="metal"`.
- **📊 Profiler Export**: Visualise traces in Chrome/Perfetto with `cp.profiler.export_chrome_trace()`.
- **⚡ Hybrid Runtime**: Rust dispatcher + C++ kernels + Objective-C++ (Metal).
- **🛡️ Cross-Platform Build**: CMake-based build system that works on Linux, macOS, and Windows.
- **📈 Efficient Stats**: Compute multiple statistics in one pass with `cp.compute_stats`.

---

## 💻 Supported Platforms

| Platform | Architecture | Accelerators | Status |
| :--- | :--- | :--- | :--- |
| **Linux** | x86_64 | AVX2, OpenBLAS | ✅ Production |
| **macOS** | Apple Silicon | **Metal**, NEON | ✅ Beta (0.2.3+) |
| **Windows** | x86_64 | AVX2 | ✅ Experimental |

---

## 🛠️ Installation

### Preferred Method (uv)
We recommend `uv` for fast, correct cross-platform installation.

```bash
# Install corepy
uv pip install corepy
```

### Fallback (pip)
Standard pip installation works if dependencies (CMake, Rust) are present.

```bash
pip install corepy
```

### From Source
```bash
git clone https://github.com/ai-foundation-software/corepy.git
cd corepy

# Using Make (Recommended)
make build && make install

# Manual
./scripts/build.sh
pip install .
```

---

## ⚡ Quick Start

### 1. Metal Acceleration (macOS)
```python
import corepy as cp

# Automatically uses Metal if available on macOS
t = cp.array([1.0, 2.0, 3.0], device="metal")
result = t.sum()
print(f"Result (GPU): {result}")
```

### 2. Performance Profiling
Stop guessing where your code is slow. Corepy has a built-in profiler.

```python
import corepy as cp

# 1. Enable profiling
cp.enable_profiling()

# 2. Run your heavy workload
x = cp.ones(1_000_000)
y = x * 3.14159
result = y.mean()

# 3. Export to Chrome Tracing format
cp.profiler.export_chrome_trace("trace.json")
```

---

## 📚 Documentation
- **[Getting Started](docs/getting-started.md)**: First steps and installation details.
- **[Metal GPU Guide](docs/05_advanced/metal_gpu.md)**: using Apple Silicon acceleration.
- **[Architecture](docs/architecture.md)**: Deep dive into the Rust/C++ hybrid runtime.
- **[Examples](examples/)**: Runnable scripts for common patterns.
- **[Contributing](docs/07_contributing/CONTRIBUTING.md)**: Build and test guide.

---

## 🤝 Stability & Roadmap
Corepy is currently **Alpha (v0.2.3)**.
- **v0.2.3**: Metal GPU Support, Robust CMake Build, Profiler Export.
- **v0.3.0**: CUDA Support and Tiled Matmul Optimization.
- **v1.0**: Stable API promise.

See [Roadmap](docs/00_overview/roadmap.md) for details.
