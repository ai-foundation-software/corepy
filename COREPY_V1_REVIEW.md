# Corepy v1: Infrastructure Review & Roadmap Alignment
---

## 1. Executive Summary & Version Scope

Corepy is being positioned as a **foundational infrastructure library**, not a user-facing application. This distinction dictates that **correctness and predictability** are strictly more important than raw throughput for v1.

### v1 vs v2 Responsibilities

| Feature Domain | **Corepy v1 (The Foundation)** | **Corepy v2 (The Performance Layer)** |
| :--- | :--- | :--- |
| **Execution Model** | **Eager & Synchronous**. Call stack is 1:1 with Python. | **Lazy & Async**. Execution graph dispatched to runtime. |
| **Parallelism** | **Zero Hidden Threads**. User sees 1 CPU core usage by default. | **Explicit Multithreading**. Work-stealing pool (Rayon/OpenMP). |
| **Hardware Support** | **CPU Only**. (SIMD allowed via compiler/bindings). | **Multi-Device**. GPU, TPU, Accelerator support. |
| **Memory Model** | **Safe Copies**. Predictable ref-counting. | **Zero-Copy Views**. Arena allocators, shared memory. |
| **Error Handling** | **Fail Fast**. Hard exceptions for invalid states. | **Resilient**. Error propagation, graceful degradation. |
| **Goal** | "It works exactly as expected, every time." | "It works faster than everything else." |

**Critical Ruling**: Any feature currently in the codebase that attempts v2 behavior (e.g., async pipeline execution, GPU detection that auto-activates) must be disabled or strictly opt-in for v1.

---

## 2. Directory & Architecture Audit

### Current State Assessment
The project follows a modern hybrid Python/Rust/C++ structure.

| Path | Component | Status | Finding |
| :--- | :--- | :--- | :--- |
| `corepy/backend` | Dispatch | ✅ **Good** | `Dispatcher` pattern is solid. `Reference` backend is MISSING. |
| `corepy/runtime` | Execution | ⚠️ **Risk** | `pipeline.py` is currently simple, but `DESIGN.md` suggests complex lazy graph plans. Keep it simple. |
| `rust/` | Scheduler | ✅ **Safe** | Minimal `pyo3` heavy lifting. No premature thread pools detected. |
| `csrc/` | Kernels | ⚠️ **Risk** | C++ kernels must NOT launch threads behind Python's back. |
| `DESIGN.md` | Doc | ❌ **Misleading** | Describes the 5-year vision (v3), confusing for v1 contributors. |
| `benchmarks/` | Infra | ❌ **Critical** | Empty/Basic. Missing rigorous warmup/measure loops. |

---

## 3. Concrete Recommendations

### A. Renaming & restructuring
1.  **Rename `DESIGN.md` → `ARCHITECTURE_VISION.md`**: Clarify that this doc represents the long-term goal, not the v1 implementation spec.
2.  **Create `v1_SPEC.md`**: A boring, precise document defining the v1 constraints (CPU-only, eager exec).

### B. Missing Components (Must Add)
1.  **Reference Backend (`corepy.backend.reference`)**:
    *   A pure Python (or NumPy-backed) implementation of every operator.
    *   **Purpose**: The "Source of Truth" for correctness testing.
    *   *If `corepy.optimized_add(a, b) != corepy.reference.add(a, b)`, Corepy is wrong.*
2.  **Benchmark Runner (`benchmarks/runner.py`)**:
    *   Must force **Garbage Collection** before runs.
    *   Must exclude compilation time (only measure execution).
    *   Must report P99 latency, not just average.

### C. Change Requests (Code Level)
1.  **Strict Device Check**: Modify `corepy/backend/dispatch.py` to hard-fail or warn if a user tries to select a GPU backend in v1, unless explicitly flagged as "Experimental".
2.  **Deterministic Randomness**: Add `corepy.setting.seed(42)` which strictly controls RNG in Python, Rust, and C++ layers simultaneously.

---

## 4. Validation Strategy

For every operator, the following **Validation Loop** applies:

1.  **Property-Based Testing (Hypothesis)**:
    *   Generate inputs: `Infinity`, `NaN`, `Empty Tensor`, `Non-Contiguous Memory`.
    *   Assert: `No Segfaults`.
2.  **The "Oracle" Test**:
    *   Compare `Corepy(x)` vs `NumPy(x)` (or `Reference(x)`).
    *   Constraint: `abs(a - b) < 1e-6` (float32) or `0` (integers).
3.  **Cross-Platform check**:
    *   CI must run the exact same seed on **Linux (AVX2)** and **macOS (ARM NEON)**.
    *   Bitwise exactness is required for v1 integer ops.

---

## 5. Performance Guidelines (v1)

We value **Latency Stability** over **Peak Throughput**.

*   **Allowed**: Memory alignment (64-byte), SIMD instructions (via compiler flags).
*   **Forbidden**: JIT Compilation at runtime (too unpredictable for v1), Thread Pools (user manages threads).
*   **Metric**: "Time to First Byte" and "Instruction Count" are better metrics than "Wall Time" for v1 ops.

---

## 6. Maintainer Checklist (Next Steps)

### Immediate (Triage)
- [ ] **Repo**: Rename `DESIGN.md` to `ARCHITECTURE_VISION.md`.
- [ ] **Infra**: Create `benchmarks/runner.py` with `warmup` and `measure` phases.
- [ ] **Code**: Implement `corepy.backend.reference` (start with one op: `add`).

### Implementation (v1 Alpha)
- [ ] **Code**: Add `corepy.settings.seed()` implementation.
- [ ] **CI**: Add a specific "Correctness" workflow that runs the Reference Oracle tests.
- [ ] **Docs**: Update `README.md` to explicitly state "CPU-First / v1" goals.
