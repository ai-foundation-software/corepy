# Corepy Performance Profiling - Complete Guide

**Version**: 0.3.0  
**Status**: Production Ready  
**Last Updated**: 2026-08-15

---

## 📖 Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Tutorial Path](#tutorial-path)
6. [Best Practices](#best-practices)
7. [Architecture](#architecture)
8. [FAQ](#faq)

---

## Introduction

### What is Performance Profiling?

Performance profiling is the process of measuring where your code spends its execution time. Instead of guessing which parts are slow, you **measure** and **know** with certainty.

### Why Add Profiling to Corepy?

**Problem**: Users don't know why their corepy code is slow
- 🤔 "Is my code slow or is the library slow?"
- 🤔 "Should I use GPU or CPU?"
- 🤔 "Which operation is the bottleneck?"

**Solution**: Built-in profiling with automatic recommendations
- ✅ See exactly where time is spent
- ✅ Get AI-powered optimization suggestions
- ✅ Compare different implementations objectively
- ✅ Detect performance regressions automatically

### Key Features

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Zero-Config** | Works out of the box | No complex setup |
| **Low Overhead** | <2% performance impact | Safe for production |
| **Automatic Tracking** | Instruments all array operations | No manual annotations |
| **Smart Recommendations** | AI-powered optimization tips | Save hours of guessing |
| **Multiple Formats** | JSON, CSV, Flamegraph, HTML | Integrate with any tool |
| **Context Management** | Profile specific sections | Focus on what matters |

---

## Quick Start

### Installation

Profiling is built into corepy. No extra installation needed!

```bash
pip install corepy-ai>=0.3.0
```

### 30-Second Example

```python
import corepy as cp

# 1. Enable profiling
cp.enable_profiling()

# 2. Run your code
data = cp.array([1.0, 2.0, 3.0, 4.0, 5.0])
result = (data * 2.0 + 5.0).mean()

# 3. See what happened
print(cp.profile_report())
```

**Output**:
```
╔══════════════════════════════════════════════════════════╗
║             COREPY PERFORMANCE REPORT                    ║
╚══════════════════════════════════════════════════════════╝

┌─────────────┬─────────┬─────────┬──────────┬──────────┐
│ Operation   │ Count   │ Avg (ms)│ Backend  │ % Total  │
├─────────────┼─────────┼─────────┼──────────┼──────────┤
│ mul         │ 1       │ 0.05    │ CPU      │ 25.0%    │
│ add         │ 1       │ 0.04    │ CPU      │ 20.0%    │
│ mean        │ 1       │ 0.11    │ CPU      │ 55.0%    │
└─────────────┴─────────┴─────────┴──────────┴──────────┘

Total Time: 0.20ms
```

That's it! You now know where your code spent its time.

---

## Core Concepts

### 1. Global Profiling

**When to use**: Quick debugging, small scripts  
**How it works**: Tracks ALL operations from enable to disable

```python
cp.enable_profiling()    # Start tracking
# ... all operations tracked ...
cp.disable_profiling()   # Stop tracking
```

**Pros**: Simple, no code changes needed  
**Cons**: Can be noisy in large applications

---

### 2. Context-Based Profiling

**When to use**: Large applications, production monitoring  
**How it works**: Only tracks operations inside `with` blocks

```python
from corepy.profiler import ProfileContext

# This is NOT profiled
setup()

# Only this is profiled
with ProfileContext("critical_loop"):
    for i in range(100):
        process(data)

# Get targeted report
report = cp.profile_report(context="critical_loop")
```

**Pros**: Focused reports, less noise  
**Cons**: Requires code annotation

---

### 3. Decorator-Based Profiling

**When to use**: Profile specific functions  
**How it works**: Decorates functions to auto-track

```python
from corepy.profiler import profile_operation

@profile_operation
def my_algorithm(data):
    # Automatically profiled!
    return data.matmul(data)
```

**Pros**: Reusable, clean code  
**Cons**: Only tracks function-level, not individual ops

---

## API Reference

### Core Functions

#### `enable_profiling()`
```python
cp.enable_profiling()
```
Enables global profiling. All array operations will be tracked.

**Returns**: None  
**Side Effects**: Starts background profiling thread  
**Overhead**: <2% when enabled, 0% when disabled

---

#### `disable_profiling()`
```python
cp.disable_profiling()
```
Disables global profiling.

**Returns**: None  
**Side Effects**: Stops tracking, data retained until cleared

---

#### `profile_report(context=None, format='table')`
```python
report = cp.profile_report(
    context="section_name",  # Optional: filter by context
    format="table"           # Options: 'table', 'json', 'compact'
)
```
Generates a performance report.

**Parameters**:
- `context` (str, optional): Filter by ProfileContext name
- `format` (str): Output format ('table', 'json', 'compact')

**Returns**: Formatted report string or dict (if format='json')

---

#### `export_profile(path, format='json')`
```python
cp.export_profile(
    path="/path/to/output.json",
    format="flamegraph"  # Options: 'json', 'csv', 'flamegraph', 'chrome'
)
```
Exports profiling data to a file.

**Parameters**:
- `path` (str): Output file path
- `format` (str): Export format
  - `'json'`: Standard JSON format
  - `'csv'`: Spreadsheet-compatible
  - `'flamegraph'`: Speedscope.app compatible
  - `'chrome'`: Chrome Tracing (chrome://tracing)

**Returns**: None (writes to file)

---

#### `clear_profile()`
```python
cp.clear_profile()
```
Clears all collected profiling data.

**Returns**: None  
**Use Case**: Reset before profiling a different section

---

#### `get_recommendations()`
```python
recs = cp.get_recommendations()
# Returns list of recommendation dicts
```
Analyzes profiling data and suggests optimizations.

**Returns**: List[Dict] with keys:
- `title` (str): Short recommendation title
- `priority` (str): 'HIGH', 'MEDIUM', 'LOW'
- `estimated_speedup` (str): e.g., "5x faster"
- `description` (str): Detailed explanation
- `code_example` (str): Before/after code snippet

---

### Context Manager

#### `ProfileContext`
```python
from corepy.profiler import ProfileContext

with ProfileContext("section_name"):
    # Only operations here are tracked
    pass
```

**Parameters**:
- `name` (str): Unique identifier for this section

**Use Cases**:
- Profile specific code blocks
- Compare different implementations
- Production monitoring of critical paths

---

### Decorator

#### `@profile_operation`
```python
from corepy.profiler import profile_operation

@profile_operation
def my_function(data):
    return data * 2
```

Automatically profiles decorated functions.

**Parameters**: None (decorator takes no args)  
**Returns**: Wrapped function that tracks execution time

---

## Tutorial Path

We've created a comprehensive tutorial series:

### 🟢 Beginner: Tutorial 01 - Profiling Basics
**Time**: 10 minutes  
**Location**: `tutorials/01_profiling_basics/`

Learn:
- How to enable/disable profiling
- How to generate reports
- How to interpret metrics
- Performance baselines

**Start**: `python tutorials/01_profiling_basics/01_enable_profiling.py`

---

### 🟡 Intermediate: Tutorial 02
**Time**: 20 minutes  
**Location**: `tutorials/02_intermediate/`

Learn:
- Context managers
- Custom decorators
- Bottleneck detection
- Applying optimization recommendations

**Start**: `python tutorials/02_intermediate/01_context_manager.py`

---

### 🔴 Advanced: Tutorial 03
**Time**: 30 minutes  
**Location**: `tutorials/03_advanced/`

Learn:
- Flamegraph analysis
- Export to external tools
- Custom baselines
- Production monitoring

**Start**: `python tutorials/03_advanced/01_flamegraph_analysis.py`

---

### 💼 Case Studies: Tutorial 04
**Time**: Variable  
**Location**: `tutorials/04_case_studies/`

Real-world examples:
- ML training loop optimization
- Data pipeline tuning
- CPU→GPU migration

---

## Best Practices

### ✅ DO

1. **Profile Before Optimizing**
   ```python
   # Don't guess - measure!
   cp.enable_profiling()
   run_code()
   report = cp.profile_report()
   ```

2. **Use Context Managers for Large Apps**
   ```python
   # Only profile what matters
   with ProfileContext("training"):
       train_model()
   ```

3. **Verify Optimizations**
   ```python
   # Always compare before/after
   with ProfileContext("before"):
       slow_version()
   
   with ProfileContext("after"):
       fast_version()
   ```

4. **Set Performance Baselines**
   ```python
   # Know what "good" looks like
   baseline = run_baseline_test()
   assert current_performance < baseline * 1.2  # Max 20% slower
   ```

---

### ❌ DON'T

1. **Don't Optimize Without Data**
   ```python
   # BAD: Guessing what's slow
   # "I think this loop is slow, let me optimize it"
   
   # GOOD: Knowing what's slow
   report = cp.profile_report()
   # "The report shows matmul takes 80% of time"
   ```

2. **Don't Profile Everything in Production**
   ```python
   # BAD: Always-on global profiling
   cp.enable_profiling()  # In production startup code
   
   # GOOD: Targeted production profiling
   if DEBUG_MODE:
       cp.enable_profiling()
   ```

3. **Don't Ignore Recommendations**
   ```python
   # BAD: Ignoring automated advice
   recs = cp.get_recommendations()  # Never looked at
   
   # GOOD: Review and apply high-priority recs
   for rec in recs:
       if rec['priority'] == 'HIGH':
           apply_recommendation(rec)
   ```

---

## Architecture

### The 3-Layer Profiling Model

Corepy's profiler follows the same 3-layer architecture as the core library:

```
┌─────────────────────────────────────────────────────────┐
│  Python Layer (corepy/profiler/)                        │
│  - User-facing API (enable, report, export)             │
│  - Report formatting and visualization                  │
│  - Context managers and decorators                      │
│  - Recommendation engine (pattern detection)            │
└──────────────────┬──────────────────────────────────────┘
                   │ PyO3 FFI
┌──────────────────▼──────────────────────────────────────┐
│  Rust Layer (rust/corepy-runtime/src/profiler/)         │
│  - Thread-safe global profiler (Mutex<Profiler>)        │
│  - Event collection (start_op, end_op)                  │
│  - Metric aggregation and analysis                      │
│  - Export format generation (JSON, CSV, Flamegraph)     │
│  - Automatic bottleneck detection                       │
└──────────────────┬──────────────────────────────────────┘
                   │ Instruments existing calls
┌──────────────────▼──────────────────────────────────────┐
│  C++ Layer (UNCHANGED)                                   │
│  - Kernels run as normal                                │
│  - Rust measures time around C++ FFI calls              │
│  - Zero impact on kernel performance                    │
└─────────────────────────────────────────────────────────┘
```

### Low-Overhead Design

**How we keep overhead <2%:**

1. **Thread-Local Buffers**: Each thread writes to its own buffer (lock-free)
2. **Batch Flushing**: Events flushed to analyzer every N operations
3. **Lazy Initialization**: Profiler only initialized when enabled
4. **Zero Cost When Disabled**: Compiled out entirely when profiling is off

---

## FAQ

### General Questions

**Q: Does profiling slow down my code?**  
A: Yes, but minimally. We target <2% overhead when enabled, 0% when disabled.

**Q: Can I use this in production?**  
A: Yes! Use context managers to profile only critical sections. Many users enable profiling for <1% of requests to monitor performance.

**Q: What happens if I forget to disable profiling?**  
A: Memory usage will grow as more events are collected. We recommend calling `cp.clear_profile()` periodically if running long-term.

---

### Technical Questions

**Q: How accurate are the timings?**  
A: Very accurate for operations >1ms. For sub-millisecond operations, there's ~0.01ms measurement uncertainty.

**Q: Can I profile GPU operations?**  
A: Yes! The profiler tracks both CPU and GPU operations and shows which backend was used.

**Q: How do I profile multi-threaded code?**  
A: Works automatically! The profiler is thread-safe and shows total time across all threads.

---

### Troubleshooting

**Q: My report is empty!**  
A: Make sure you called `cp.enable_profiling()` before running operations, or used a `ProfileContext`.

**Q: Recommendations seem wrong**  
A: Recommendations are heuristic-based. Use your judgment! If a suggestion doesn't make sense for your use case, ignore it.

**Q: Export failed with "file not found"**  
A: Ensure the output directory exists. Corepy doesn't create parent directories automatically.

---

## Contributing

Want to improve the profiler? See [CONTRIBUTING.md](../CONTRIBUTING.md).

Priority areas:
- [ ] ML-based recommendation engine
- [ ] Distributed profiling (multi-node)
- [ ] Memory profiling (in addition to time)
- [ ] GPU kernel-level profiling

---

## License

Apache 2.0 - Same as corepy core

---

## Changelog

### v0.2.0 (2026-01-27)
- ✨ **NEW**: Complete profiling system
- ✨ **NEW**: Automatic optimization recommendations
- ✨ **NEW**: Flamegraph export
- ✨ **NEW**: Context-based profiling
- ✨ **NEW**: Comprehensive tutorial series

---

**Built with ❤️ by the Corepy Team**
