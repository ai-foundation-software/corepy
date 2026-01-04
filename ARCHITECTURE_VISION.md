# Corepy: Architectural Design & Implementation Strategy

**Status**: Draft
**Author**: Systems Architect (Antigravity)
**Date**: 2025-12-14
**Target Audience**: Core Contributors, Architects, Maintainers

---

## 1. Executive Summary & Recommended Tech Stack

Corepy is designed as a unified, high-performance runtime foundation for the next decade of Python computing. It combines the ease of Python with the raw speed of C++ kernels and the safety/concurrency of Rust.

### Recommended Tech Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Public API** | **Python 3.10+** | Ubiquity, ecosystem compatibility, extensive type hinting. |
| **Compute Kernels** | **C++20** | Unmatched compiler optimization for SIMD (AVX-512/NEON), mature math libraries (Eigen, BLAS), explicit memory layout control. |
| **Runtime & Safety** | **Rust (Stable)** | Memory safety without GC, fearless concurrency for the task scheduler, robust plugin system data races prevention. |
| **Python Bindings** | **pybind11 (C++)** <br> **PyO3 (Rust)** | Best-in-class tools for each language. `naming compatibility` maintained via strict API wrapping. |
| **Build System** | **CMake** + **Scikit-build-core** | Standard for C++ integration. Handles complex cross-compilation and linking. |
| **Package Manager** | **uv** | Modern, extremely fast Python package and environment management. |
| **Distribution** | **cibuildwheel** | Standard for producing manylinux/musllinux wheels. |
| **Interchange** | **Apache Arrow** (ABI) | Zero-copy data sharing with the wider ecosystem (pandas, Polars, DuckDB). |

---

## 2. Core Languages & Responsibilities

The "Tri-Language" architecture is chosen to maximize the strengths of each language while mitigating their weaknesses.

### 2.1 Python (The User Interface)
*   **Role**: Public API, configuration, high-level graph definition, method dispatch.
*   **Responsibilities**:
    *   Defining the `Schema` and `DataFrame` / `Tensor` APIs.
    *   Constructing the "Lazy Execution Graph" (the user describes *what* to do, not *how*).
    *   Python-side error handling and formatting.
    *   Integration with Pydantic for validation.

### 2.2 C++ (The Compute Engine)
*   **Role**: Raw number crunching, hardware-specific optimizations, hot loops.
*   **Rationale**: C++ is still the king of compiler vectorization and has the richest ecosystem of numerical linear algebra libraries (e.g., Eigen, xsimd).
*   **Responsibilities**:
    *   **Strided Array Operations**: Basic arithmetic, reductions, broadcasting.
    *   **SIMD Intrinsics**: Hand-tuned AVX-512 / NEON implementations where compilers fail.
    *   **GPU Kernels**: CUDA/Metal integrations often have the best support in C++.
    *   **Memory Management**: Custom allocators (pool, arena) to minimize page faults.

### 2.3 Rust (The System Architect)
*   **Role**: System orchestration, thread scheduling, safety, I/O.
*   **Rationale**: Rust’s ownership model prevents header-induced race conditions which are plague in complex C++ multi-threading.
*   **Responsibilities**:
    *   **Task Scheduler**: A work-stealing scheduler (built on top of `tokio` or `rayon` logic) to execute the dependency graph.
    *   **Plugin System**: Loading dynamic shared objects safely.
    *   **IO Layer**: Async file reading (Parquet/IPC) and network streaming.
    *   **Global State Management**: Keeping track of resources, logging, and metrics.

### Boundary Enforcement
*   **FFI Layer**: C++ exposes a stable C ABI (extern "C"). Rust calls C++ via `bindgen`. Python calls both via native extensions.
*   **Testing**: "Sandwich" tests—Python calls Rust which orchestrates C++ kernels, verifying the full stack.

---

## 3. Python Integration Layer

We prioritize **Stable APIs** over Stable ABIs for the initial years, moving towards limited ABI stability later.

*   **Bindings**:
    *   **C++ Kernels** $\rightarrow$ `pybind11`: Chosen for its seamless mapping of C++ STL containers to Python types and mature handling of NumPy buffer protocols.
    *   **Rust Runtime** $\rightarrow$ `PyO3` + `maturin`: PyO3 offers superior safety and ergonomics for creating Python classes from Rust structs.
*   **Distribution Strategy**:
    *   Self-contained `wheels` ensuring `pip install corepy` "just works".
    *   Static linking of C++/Rust dependencies where possible to avoid `LD_LIBRARY_PATH` hell.
    *   **Versioning**: Semantic Versioning (SemVer).
        *   `0.x`: Rapid evolution.
        *   `1.0`: API stability guarantee.
        *   **Deprecation Policy**: Two-minor-version warning period.

---

## 4. Hardware Utilization Strategy

Corepy is "Hardware Aware" by default, detecting capabilities at runtime.

### 4.1 CPU Strategy (The Reliable Backbone)
*   **Execution**:
    *   **Small Ops**: Executed immediately on the calling thread (low overhead).
    *   **Large Ops**: dispatched to a Rust-managed `Rayon` thread pool. Workers are pinned to cores to reduce context switching.
*   **SIMD**:
    *   Runtime dispatch (CPUID check) selects the hottest kernel path (Generic $\rightarrow$ AVX2 $\rightarrow$ AVX-512).
    *   Use `xsimd` or `Highway` in C++ to write vector-agnostic code that compiles to multiple instruction sets.

### 4.2 GPU Strategy (Phased Rollout)
*   **Phase 1 (NVIDIA & Apple)**:
    *   Optional `corepy-cuda` and `corepy-metal` extras.
    *   Explicit `.to_device("cuda:0")` API moves data.
*   **Phase 2 (Unified)**:
    *   "Backend" abstraction. Usage: `corepy.set_backend("gpu")`.
*   **Phase 3 (Fallback Chain)**:
    *   Attempt GPU $\rightarrow$ Fallback to SIMD CPU $\rightarrow$ Fallback to Scalar CPU.
    *   Configurable via `COREPY_DEVICE_PRIORITY=gpu,cpu`.

### 4.3 Memory Layout
*   **Zero-Copy**: Adhere to the Apache Arrow memory specification where possible for efficient interchange.
*   **Alignment**: 64-byte alignment for cache lines and AVX-512 compatibility.
*   **NUMA**: (Future) Thread affinity policies to process data on the generic CPU node where it resides.

---

## 5. Cross-Platform Compatibility

We commit to "Tier 1" support for major OS/Arch combinations.

| Platform | Arch | Notes |
| :--- | :--- | :--- |
| **Linux (glibc)** | x86_64 | Manylinux_2_28 compliant. |
| **Linux (glibc)** | aarch64 | Server ARM (AWS Graviton). |
| **Linux (musl)** | x86_64 | Alpine support for lightweight containers. |
| **macOS** | arm64 | Apple Silicon (M1/M2/M3). Optimized with NEON. |
| **macOS** | x86_64 | Legacy Intel Mac support (Maintenance mode). |
| **Windows** | x86_64 | MSVC toolchain integration. |

*   **Build Pitfalls**:
    *   **Windows**: path length limits and lack of some POSIX signals (handled by Rust standard library abstractions).
    *   **Musl**: Dynamic linking issues; prefer static linking for Rust/C++ runtime deps.

---

## 6. Build & Packaging Stack

We treat the build system as a product feature.

*   **Orchestrator**: `scikit-build-core`.
    *   *Why?* It acts as the bridge between standard pip builds (`pyproject.toml`) and CMake, allowing us to build the C++ parts seamlessly.
*   **Rust Bridge**: `maturin`.
    *   *Why?* Proven robustness for Rust-Python extensions. We may effectively have two extensions: `_corepy_cpp` (via scikit-build) and `_corepy_rust` (via maturin), or invoke Cargo via CMake (Corrosion). *Decision*: **Hybrid Build**. `scikit-build-core` invoking CMake, which uses `Corrosion` to build the Rust crates and link them into a single extension library. This simplifies distribution to a single shared object if possible, or two distinct `.so`/`.pyd` files wrapped by Python.
*   **Toolchain**: `uv`.
    *   *Why?* Replaces pip/poetry/venv with a single, ultra-fast tool for dependency resolution and venv management.

---

## 7. Project Structure (Monorepo)

```text
/corepy/
├── .github/              # CI/CD workflows
├── bindings/             # Language bindings
│   ├── python/           # Python native extension code
│   └── c/                # C ABI headers for FFI
├── cmake/                # CMake modules and helper scripts
├── corepy/               # The actual Python package source
│   ├── __init__.py
│   ├── ops/              # Python operator definitions
│   └── schema/           # Pydantic schema integrations
├── csrc/                 # C++ Source (Compute Kernels)
│   ├── include/
│   ├── kernels/          # SIMD implementations
│   └── CMakeLists.txt
├── rust/                 # Rust Source (Runtime System)
│   ├── corepy-runtime/   # Main scheduler crate
│   └── Cargo.toml
├── tools/                # Build and dev scripts
├── tests/                # Testing Framework
│   ├── bench/            # pytest-benchmark & google-benchmark
│   ├── fuzz/             # Hypothesis strategies
│   └── unit/
├── pyproject.toml        # Unified build config
├── CMakeLists.txt        # Master CMake
└── README.md
```

---

## 8. Performance & Testing Strategy

**Philosophy**: "Correctness First, Performance a close Second."

### Testing
*   **Property-Based Testing (`hypothesis`)**: Generate random DataFrames/Tensors and compare results against a "Golden Reference" (simple, slow Python implementation or NumPy).
*   **FFI Boundary Tests**: Verify that passing invalid pointers, wrong types, or nulls from Python is caught safely by Rust/C++ before segfaulting.
*   **Sanitizers**: Run CI with ASAN (AddressSanitizer) and TSAN (ThreadSanitizer) to catch memory leaks and race conditions in C++/Rust.

### Benchmarking
*   **Continuous Benchmarking**: Run specific micro-benchmarks on every PR to detect regressions.
*   **Tools**:
    *   Python: `pytest-benchmark` for end-to-end overhead.
    *   C++: `google-benchmark` for kernel throughput.
    *   Rust: `criterion` for scheduler latency.
*   **Profile**: Using `perf` (Linux) and `Instruments` (macOS) to visualize flamegraphs.

---

## 9. Developer Experience (DX)

*   **Type Hints**: 100% type coverage in Python. No `Any` allowed in public APIs.
*   **Tracing**: Built-in support for `OpenTelemetry` or a lightweight chrome-tracing compatible output to visualize execution graphs.
*   **Debug Mode**: `COREPY_DEBUG=1` enables extra bounds checking, logging, and disables some async optimizations for readable stack traces.
*   **Pre-commit**:
    *   `ruff` (Python linting/formatting).
    *   `clang-format` (C++).
    *   `cargo fmt` (Rust).

---

## 10. Governance & Roadmap

### Open Source Governance
*   **License**: Apache 2.0 (Industry standard, patent-friendly).
*   **Contribution Workflow**: RFC (Request for Comments) process required for any major API change or new sub-system.
*   **Maintainers**: "Benevolent Dictatorship" transitioning to a Steering Committee after version 1.0.

### Long-Term Scalability
*   **Year 1**: Foundation. CPU-only, rigid schema, basic operators.
*   **Year 3**: Ecosystem. GPU backend, distributed plugin system, dataframe API parity.
*   **Year 5+**: The Platform. Serving as the backend for other libraries (e.g., standardizing the NumPy ABI replacement).

---

## 11. Example Use Cases

### AI Preprocessing Pipeline
```python
import corepy as cp
from corepy.schema import Tensor, Float32

# Define a lazy graph execution pipeline
def preprocess(image_batch: cp.Tensor) -> cp.Tensor:
    # Operations are hardware-aware and fused where possible
    normalized = (image_batch - 0.485) / 0.229
    # Handled by C++ SIMD kernels
    resized = cp.vision.resize(normalized, (224, 224))
    return resized

# Execution happens on Rust-managed thread pool
batch = cp.read_images("./data/*.jpg", parallel=True)
result = preprocess(batch).compute(device="auto")
```

### High-Frequency Data Engineering
```python
import corepy.data as cpd

# Zero-copy load from Arrow IPC
df = cpd.read_ipc("trade_data.arrow")

# Rust scheduler handles the windowing concurrency safely
features = (
    df.select("symbol", "price", "timestamp")
    .group_by("symbol")
    .rolling("1s")
    .agg([
        cp.mean("price").alias("vwap"),
        cp.std("price").alias("volatility")
    ])
)

# Output is ready for ML ingestion
features.write_parquet("features.parquet")
```
