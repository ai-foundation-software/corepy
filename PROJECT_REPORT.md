# Corepy Project Audit & Documentation

## 1. Executive Summary

Corepy is a research-grade, infrastructure-level runtime designed to unify data, computation, and AI workflows. It leverages a novel "Tri-Language" architecture (Python for API, C++ for Compute, Rust for Orchestration) to solve the "Two-Language Problem" (Ease of use vs. Performance) without the typical safety compromises of pure C++ extensions. It is designed for longevity (5+ years), prioritizing correctness, reproducibility, and transparent hardware acceleration over short-term benchmarks or marketing hype.

## 2. Purpose

The core purpose of Corepy is to provide a **safe, performant, and hardware-aware foundation** for high-performance Python applications.

*   **Core Problem Solved**: The fragmentation and safety risks in existing high-performance Python libraries. Current tools often force users to choose between safety (pure Python/Rust) and raw speed (C++/CUDA), or struggle with the GIL in multi-threaded contexts.
*   **Why Existing Tools fail**: NumPy is largely single-threaded and lacks native GPU support. PyTorch/TensorFlow are massive dependencies optimized for Deep Learning, not general-purpose computing, and are difficult to extend.
*   **Guarantees**:
    *   **Memory Safety**: Enforced by the Rust runtime properties at the scheduler level.
    *   **Correctness**: Mathematical precision and rigorous testing (property-based) take precedence over speed.
    *   **Hardware Transparency**: Automatic, safe dispatch to SIMD (CPU) or GPU backends without code changes.
*   **Non-Goals**:
    *   **Being a "web framework"** or general-purpose application builder.
    *   **Supporting legacy Python versions (<3.10)** or obsoleted hardware.
    *   **Maximizing single-threaded scalar performance** (pure C is better for that).

## 3. Motivation

**Real-world Pain Points:**
*   **The "GIL Wall"**: Data pipelines stalling because Python threads cannot execute compute-bound tasks in parallel efficiently.
*   **"Segfault Hell"**: C++ extensions that crash the interpreter on invalid input, making debugging in production a nightmare.
*   **Deployment Fragility**: The difficulty of shipping hybrid Python/C++ applications that work reliably across Linux (glibc/musl), macOS (Universal), and Windows.

**Design Philosophy:**
*   **Correctness > Speed**: A fast wrong answer is useless. We use Rust to ensure that concurrency bugs are caught at compile time where possible.
*   **Reproducibility**: Experiments must be repeatable. Random number generation and scheduling must be deterministic when configured.
*   **Device-Aware, not Device-Dependent**: The code should run on a laptop (CPU) exactly as it runs on a cluster (GPU), just slower.

## 4. Use Cases

### Primary Use Cases (Intended)
1.  **AI Preprocessing Pipelines**:
    *   **User**: ML Engineers / Research Scientists.
    *   **Workflow**: Loading terabytes of distinct image/text data, applying layout transformations, and feeding them into a Training Loop.
    *   **Why Corepy**: Rust-based async loading prevents blocking the training loop; C++ SIMD handles the resize/crop bottlenecks efficiently.
2.  **High-Frequency Data Engineering**:
    *   **User**: Quantitative Developers / Data Engineers.
    *   **Workflow**: Ingesting Arrow IPC streams, computing rolling window statistics, and normalizing data in real-time.
    *   **Why Corepy**: Zero-copy Arrow integration and safe concurrency allow maximizing core usage without race conditions.

### Secondary Use Cases (Possible)
1.  **Scientific Simulation**:
    *   **User**: Physicists / Biologists.
    *   **Workflow**: Grid-based simulations or tensor algebra that fits in memory.
    *   **Why Corepy**: Easier to write than C++, faster than NumPy.

### Non-Goals (Explicitly Unsupported)
1.  **Simple CRUD / Web Apps**: Use Django/FastAPI. Corepy overhead is unjustified.
2.  **Small Data Scripting**: For lists of <1000 items, standard Python `list` or `dict` is faster and simpler.
3.  **Legacy Systems**: Systems that cannot run Python 3.10+ or lack AVX2/NEON support (unless using a fallback generic build).

## 5. Directory & File Structure Review

| Directory | Responsibility | Status | Notes |
| :--- | :--- | :--- | :--- |
| `corepy/` | Python User Interface & API | ✅ Good | The root package. Clean separation of concerns. |
| `corepy/backend/` | Hardware abstraction & selection | ✅ Good | Correct placement for device logic. |
| `corepy/ops/` | Operator definitions | ✅ Good | Keep implementation separate from definition here. |
| `csrc/` | C++ Compute Kernels | ✅ Good | Standard location for extension code. |
| `rust/` | Rust Runtime & Scheduler | ✅ Good | Clearly separated from C++. |
| `bindings/` | FFI (Pybind11/PyO3) | ✅ Good | Excellent practice to isolate glue code from logic. |
| `examples/` | User-facing demos | ✅ Good | Useful for onboarding. |
| `docs/` | Documentation | ⚠️ Needs Work | Structure is flat; needs hierarchy. |
| `tests/` | QA | ✅ Good | Ensure "sandwich" tests (Py->Rust->C++) are here. |

## 6. File-Level Recommendations

*   **`DESIGN.md`**: This is a critical artifact. **Recommendation**: Move to `docs/design/architecture.md` to make it part of the official docs site, rather than a loose file in root.
*   **`corepy/compute/`**: Ensure this doesn't duplicate `corepy/ops/`. If `ops` defines the API, `compute` should perhaps map to the low-level dispatch. Clarify strict boundary.
*   **`pyproject.toml`**: Ensure "classifiers" are accurate. Currently implies "Alpha".
*   **New Files Needed**:
    *   `docs/BENCHMARKS.md`: Tracking performance over time.
    *   `CONTRIBUTING_GUIDE.md`: Detailed setup for the hybrid environment (compiling Rust+C++).

## 7. Documentation Structure Recommendation

Proposed `docs/` layout:

```text
docs/
├── index.md                # Landing page
├── install.md              # Detailed install guide
├── tutorial/               # "Zero to Hero"
│   ├── quickstart.md
│   └── concepts.md
├── reference/              # API Docs (autogenerated)
│   ├── corepy.md
│   └── corepy.backend.md
├── topics/                 # Deep dives
│   ├── memory_management.md
│   └── multithreading.md
├── design/                 # Architecture & meta-docs
│   ├── architecture.md     # (Old DESIGN.md)
│   └── backend_selection.md
└── contributing.md         # Dev guide
```

## 8. Maintainability Check

*   **Modularity**: High. The separation of Compute (C++) and Runtime (Rust) is a strong architectural decision that enforces clean boundaries.
*   **Tech Debt Risk**: **High Complexity**. Requiring developers to know Python, C++, *and* Rust is a high barrier to entry.
    *   *Mitigation*: Strict CI checks, very detailed "Internals" documentation, and encapsulating language-specific logic so contributors can focus on just one layer.
*   **Dependency Boundaries**: Good. `pyproject.toml` is lean.

## 9. Next Improvement Path

1.  **Move `DESIGN.md`** to `docs/design/architecture.md`.
2.  **Formalize the Backend Protocol**: Ensure `corepy/backend` has abstract base classes that enforce the contract for new devices.
3.  **Expand CI**: Add a "Sanitizer" workflow (ASAN/TSAN) as mentioned in the design doc, if not present.
