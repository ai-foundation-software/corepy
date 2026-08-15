# AGENT.md — Developer & AI Agent Guide for Corepy

> **Repository:** `ai-foundation-software/corepy`  
> **Target Audience:** AI Coding Assistants (Antigravity, Cursor, Claude, etc.) & Corepy Contributors  
> **Project Purpose:** High-Performance Array & DataFrame Runtime for Python powered by a Rust Core Engine.

---

## 1. Project Architecture Overview

Corepy is a hybrid Python/Rust array runtime designed for high-performance computing, scientific simulations, and AI foundations.

```
corepy/
├── corepy/                       # Python API Layer
│   ├── __init__.py               # Top-level exports (cp.array, cp.read_csv, etc.)
│   ├── array.py                  # CorePy NDArray class & UFUNC operations
│   ├── dataframe.py              # DataFrame runtime (relational pushdown)
│   ├── series.py                 # Series structure & column operations
│   ├── buffer_pool.py            # Array memory allocation reuse & LRU caching
│   ├── matmul.py                 # Matrix multiplication & BLAS dispatch logic
│   ├── broadcasting.py           # NumPy-compatible shape broadcasting rules
│   ├── lazy/                     # Expression tree construction & kernel fusion
│   ├── compute/                  # Compute engines & device backends
│   ├── backend/                  # CPU/Metal/CUDA backend detection & dispatch
│   └── profiler/                 # Chrome trace exporter & performance tracking
├── rust/
│   ├── Cargo.toml                # Workspace Cargo configuration
│   └── core/                     # Rust Core Engine crate (`_corepy_rust`)
│       ├── Cargo.toml            # Maturin / PyO3 configuration
│       └── src/
│           ├── lib.rs            # PyO3 module definition & entrypoint
│           ├── array/            # Rust tensor storage & SIMD memory buffers
│           ├── backend/          # BLAS/SIMD hardware detection & dispatch
│           ├── dataframe/        # Relational algebra & columnar operations
│           ├── ffi/              # C/Python C-API FFI wrappers
│           ├── linalg/           # SIMD accelerated BLAS & Matmul routines
│           ├── ops/              # UFUNC core math operations (AVX2/NEON)
│           ├── profiler/         # Nanosecond profile trace recorder
│           └── scheduler/        # Rayon parallel executor & thread pools
├── scripts/                      # Build automation & environment verifiers
│   ├── build.sh                  # Development build runner
│   ├── rebuild.sh                # Clean & rebuild script
│   └── verify_install.py         # Runtime integrity validator
├── tests/                        # Pytest suite for Python API & Rust integration
├── benchmarks/                   # Micro-benchmarks for ops, matmul & dataframe
└── docs/                         # Architecture guides & API documentation
```

---

## 2. Environment Setup & Dependency Management

Corepy uses **`uv`** for Python package management and **`cargo`** for the Rust toolchain.

### Prerequisites & Initialization
```bash
# Install Python dependencies into local .venv using uv
uv sync --all-extras --group dev --no-install-project
source .venv/bin/activate
```

---

## 3. Build & Development Workflow

> ⚠️ **CRITICAL RULE FOR AI AGENTS:**  
> Corepy relies on native Rust C extensions (`_corepy_rust`). Whenever you modify Rust code inside `rust/core/src/`, you **MUST** rebuild the Rust extension before running pytest or executing Python scripts.

### Build Commands
```bash
# Build the Rust extension in development mode (Recommended)
make build
# OR directly via maturin:
uv run maturin develop --manifest-path rust/core/Cargo.toml

# Clean rebuild of Rust extension & Python artifacts
make rebuild

# Build release distribution wheel
make wheel
```

### Verification & Compatibility Check
```bash
# Check compiler toolchains and environment libraries
make check-compatibility

# Verify native module import & system BLAS detection
make check
```

---

## 4. Testing, Linting & Quality Control

Always verify code changes with the project test suite and linter.

### Running Tests
```bash
# Run complete test suite with coverage
make test
# OR
uv run pytest tests/ --cov=corepy --cov-report=term -v

# Run targeted test file
uv run pytest tests/test_array.py
```

### Code Formatting & Linting
```bash
# Format Python code (Ruff)
make format
# OR: uv run ruff format .

# Lint Python code (Ruff)
make lint
# OR: uv run --no-sync ruff check . --fix

# Check Rust code
make rust-check
# OR: uv run --no-sync cargo check --manifest-path rust/core/Cargo.toml

# Lint Rust code (Clippy)
make rust-lint

# Format Rust code
make rust-fmt
```

---

## 5. Agent Implementation Rules & Guidelines

When working in this repository, follow these core principles:

1. **Rust-First Execution for Heavy Compute:**  
   Heavy mathematical operations, array allocation, SIMD loops, and relational joins must be handled in `rust/core/src/`. Python wrapper layers in `corepy/` should focus on API convenience, type annotations, and shape validation.

2. **Always Use `uv` for Python Commands:**  
   Prefix Python tool executions with `uv run` (e.g. `uv run pytest`, `uv run ruff`). Never call system `pip` directly.

3. **Propagate Exceptions Cleanly:**  
   Never swallow runtime exceptions silently. In Rust PyO3 code, convert errors into standard `PyValueError`, `PyTypeError`, or `PyRuntimeError` with clear diagnostic messages.

4. **BLAS Backend Hierarchy:**  
   Corepy auto-detects system BLAS libraries in the following order of preference:
   1. Intel MKL (`libmkl_rt.so`)
   2. AMD AOCL (`libblis-mt.so`)
   3. Apple Accelerate (macOS `aarch64`)
   4. OpenBLAS (`libopenblas.so`)
   5. Pure-Rust Fallback (Rayon parallelized)  
   Respect `COREPY_BACKEND` environment variable overrides when debugging backends.

5. **Memory & Buffer Pool Management:**  
   Avoid allocating new uninitialized memory when reusable buffers are available in `buffer_pool.py`.

6. **Preserve Compatibility & Documentation:**  
   Keep existing docstrings, maintain type annotations (`ArrayLike`, `DType`), and update tests when modifying public API methods in `cp.*`.

---

## 6. Comprehensive CI Pipeline

Run the master CI check before declaring success on major task completions:
```bash
make ci
```
*(Runs format check, Python linting, Rust format, Rust check, Rust clippy, multi-target cross checks, development build, and unit tests.)*
