# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-08-16

### Fixed
- **PyPI Release**: Version bump to re-publish release artifacts to PyPI.

## [0.3.0] - 2026-03-22

### Added — Rust-Native Runtime

- **Pure Maturin Build**: Migrated from C++/CMake to pure Rust via `maturin` + PyO3. All math logic now lives in Rust with zero C++ dependency.
- **`CoreArray`**: Rust-backed array type as the strict backing store for all operations. Pure Python and NumPy fallbacks removed.
- **Python 3.14 Support**: Bumped supported Python versions to 3.10–3.14.

### Added — UFUNC CORE-50 & Linalg

- **50+ Universal Functions**: Complete NumPy-compatible ufunc library (Trig, Hyperbolic, Exp/Log, Rounding, Bitwise, Reductions, Stacking).
- **Advanced Linalg**: Verified and tested `linalg.inv`, `linalg.det`, and `linalg.norm` for matrices $n > 2$ using the `faer` backend.
- **Ufunc Engine**: Centralized dispatch via `corepy.ops.ufunc_engine` with Rust-first execution.

### Added — Lazy Evaluation

- **`cp.lazy()` Context Manager**: Builds expression trees instead of executing immediately. Call `.compute()` to materialize.
- **Fusion-Ready IR**: Lazy arrays record operations for future optimization and kernel fusion.

### Added — Adaptive CPU Backend

- **Runtime Hardware Awareness**: Vendor detection (Intel/AMD/Apple) and SIMD optimization (AVX2, AVX512, NEON, AMX).
- **Intelligent Dispatch**: Priority-ordered BLAS selection (MKL → AOCL → Accelerate → OpenBLAS → Pure-Rust).
- **Adaptive Threading**: Thread policy that avoids oversubscription based on matrix size and CPU topology.

### Added — Data & Random

- **DataFrame & Series**: Rust-backed `DataFrame` with CSV I/O, parallelized `groupby`, and relational operations.
- **Random Module**: Parallelized PCG64 and Xoshiro generators (`rand`, `randn`, `randint`).

### Added — CI, Build & Docs

- **Automated Performance Tracking**: Integrated `scripts/benchmark.py` into GitHub Actions CI for regression detection.
- **Self-Contained Wheels**: Robust BLAS DLL bundling for Linux and improved Windows discovery.
- **Advanced Documentation**: New Windows Installation Guide and Testing Strategy rationale.
- **Project Sanitization**: Consolidated multi-language utilities into `scripts/` and removed legacy root-level artifacts.

### Changed

- `matmul_f32_cpu_dispatch` uses vendor-aware backend selection and adaptive thread policy.
- `build.rs` supports feature-gated MKL, AOCL, and robust Windows OpenBLAS detection.
- Removed all legacy C++/CMake code and root-level housekeeping scripts.
- NumPy is now strictly an optional benchmarking dependency.

### Fixed

- Linalg precision and shape consistency for multi-dimensional matrix operations.
- Windows CI `ImportError: DLL load failed` — resolved via proactive DLL pre-loading.
- Metal framework linking on macOS and backend selector test mocks.

## [0.2.4] - 2026-02-08

### Added
- **CI Local Simulation**: New `make ci` target to run full Python/Rust checks locally.
- **Improved Metal Build**: Automatic framework linking and GPU verification.

### Changed
- **Dependency Pinning**: Standardized on stable versions for `numpy`, `pydantic`, `pydantic-core`, and `pybind11`.
- **Package Rename**: Renamed PyPI package to `corepy-ai` (import as `corepy`).
- **Script Modernization**: All scripts now strictly use `uv`.

### Fixed
- **Coverage C Tracer**: Resolved `CoverageWarning` by ensuring C extensions are correctly installed.
- **Rust Lints**: Fixed identity operations and unnecessary returns in backend kernels.


## [0.2.3] - 2026-02-06

### Added
- **Metal GPU Support (macOS)**: Native `sum`, `mean`, and `matmul` kernels for Apple Silicon (M1/M2/M3). Enable with `device="metal"`.
- **Profiler Export**: New `export_chrome_trace` to visualize performance in `chrome://tracing`.
- **Zero-Copy Strided Views**: `BufferView` now supports non-contiguous memory layouts without data duplication.
- **Robust Build System**: New `csrc/CMakeLists.txt` with cross-platform support (Linux/macOS/Windows) and optional OpenBLAS detection.
- **Root Makefile**: Added `make build`, `make test`, `make clean` for standardized developer workflows.

### Changed
- **Installation**: Recommended `uv` for faster, distinct-platform installation.
- **OpenBLAS**: Now optional on all platforms; falls back to generic C++ kernels if missing.

## [0.2.1] - 2026-01-30

### Added
- **Optimization**: CPU-optimized `matmul` dispatch in `corepy.tensor` using C++ backend.
- **Tutorials**: Comprehensive guides for profiling, optimization, and advanced usage in `tutorials/`.
- **Scripts**: Moved root housekeeping scripts to `benchmarks/` and `examples/` for cleaner structure.

### Infrastructure
- **Build System**: Modernized `setup_dev.sh` and CI workflows (`.github/workflows/`) to use `uv` for dependency management.
- **Cleanup**: Removed accidental inclusion of `rust/target` build artifacts from git history.

## [0.2.0] - 2026-01-04

### Added
- **Profiling System**:
  - Zero-config profiling with `cp.enable_profiling()`.
  - Context manager `ProfileContext` for targeted profiling.
  - Decorator `@profile_operation` for custom functions.
  - Automatic bottleneck detection and optimization recommendations.
  - Support for JSON, CSV, and Flamegraph export formats.
- Complete `tutorials/` series for learning Corepy.
- `Table` data container prototype.
- `Schema` definition system.
- `Pipeline` execution engine.

### Infrastructure (2026-01-30)
- **Documentation**: Added comprehensive documentation structure (docs/00-08)
- **Development Tools**:
  - Development setup guide (DEVELOPMENT.md)
  - Automated environment setup script (setup_dev.sh)
  - Directory README files (benchmarks/, examples/, tests/)
- **Build System**:
  - Migrated to uv for dependency management
  - Added maturin to development dependencies
  - Enhanced .gitignore with OS-specific and temporary file rules
- **Project Structure**: Archived historical reports to docs/archive/

### Fixed
- Memory safety issues with non-contiguous arrays
- Build script portability across platforms
