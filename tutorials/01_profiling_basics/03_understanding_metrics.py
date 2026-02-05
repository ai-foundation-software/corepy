"""
Tutorial 01-3: Understanding Performance Metrics

WHAT: Deep dive into profiling metrics and what they mean
WHY: Raw numbers are useless without context and interpretation
HOW: Learn to compare metrics against baselines and identify real issues

Expected Time: 5 minutes
"""

import corepy as cp

# ============================================================================
# INTRODUCTION: What Makes an Operation "Slow"?
# ============================================================================
print("=" * 70)
print("🤔 WHAT MAKES AN OPERATION 'SLOW'?")
print("=" * 70)
print("""
This is the wrong question! Instead ask:
  ✅ "Is this operation slower than expected for THIS data size?"
  ✅ "Is this operation taking a disproportionate amount of total time?"
  ✅ "Could this operation run faster on different hardware?"

Let's learn how to answer these questions!
""")


# ============================================================================
# EXAMPLE 1: Comparing Small vs. Large Data
# ============================================================================
print("=" * 70)
print("EXAMPLE 1: Data Size Matters!")
print("=" * 70)

cp.enable_profiling()

# Small data (fast)
small_data = cp.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # 5 elements
result_small = small_data.sum()
print(f"Small data sum: {result_small}")

# Large data (slower, but that's expected)
large_data = cp.tensor([float(i) for i in range(10000)])  # 10,000 elements
result_large = large_data.sum()
print(f"Large data sum: {result_large}")

print("\n📊 Report for different data sizes:")
print(cp.profile_report())

print("""
💡 INSIGHT:
The large data operation takes longer, but that's EXPECTED!
What matters is:
  - Is it proportionally slower? (10000 elements ≈ 2000x more data)
  - Is the time per element roughly constant?

If processing 2000x more data takes 10000x longer, something's wrong!
""")

cp.clear_profile()


# ============================================================================
# EXAMPLE 2: Understanding "% Total Time"
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: The '% Total' Column is Your Friend")
print("=" * 70)

cp.enable_profiling()

data = cp.tensor([float(i) for i in range(1000)])

# Quick operations
for _ in range(100):
    temp = data + 1.0  # Very fast, called 100 times

# One slow operation
result = data.matmul(data)  # Matrix multiply on vector - slower

print("📊 Report showing % Total time:")
print(cp.profile_report())

print("""
💡 INSIGHT:
Even though 'add' was called 100 times, 'matmul' might take MORE total time!

The '% Total' column shows:
  ✅ Which operations dominate your runtime
  ✅ Where optimization efforts should focus

RULE OF THUMB:
  - Operations taking >50% of time → High priority to optimize
  - Operations taking 10-50% → Medium priority
  - Operations taking <10% → Don't waste time optimizing these!
""")

cp.clear_profile()


# ============================================================================
# EXAMPLE 3: Performance Baselines
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Setting Performance Baselines")
print("=" * 70)

print("""
WHAT IS A BASELINE?
A baseline is your expected performance for a given operation and data size.

HOW TO CREATE BASELINES:
1. Profile on known-good hardware with typical data
2. Record the timings
3. Compare future runs against these baselines
4. Flag operations that are >2x slower than baseline

EXAMPLE BASELINES (CPU, 1000 elements):
""")

baselines = {
    "add": "0.01 ms",
    "mul": "0.01 ms",
    "sum": "0.05 ms",
    "mean": "0.08 ms",
    "matmul": "0.50 ms",
}

for op, time in baselines.items():
    print(f"  {op:10s} : {time}")

print("""
If your profiling shows:
  ✅ add:  0.01 ms → Normal (matches baseline)
  ❌ sum:  5.00 ms → SLOW! (100x slower than expected)
  
The slow operation is your optimization target!
""")


# ============================================================================
# EXAMPLE 4: Backend Selection Impact
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Backend Matters (CPU vs GPU)")
print("=" * 70)

print("""
The 'Backend' column shows WHERE your operation ran:
  - CPU: Central Processing Unit (your computer's main processor)
  - GPU: Graphics Processing Unit (much faster for parallel work)

WHEN TO USE EACH:

CPU is faster for:
  ✅ Small data (< 1000 elements)
  ✅ Sequential operations
  ✅ Simple arithmetic

GPU is faster for:
  ✅ Large data (> 10,000 elements)
  ✅ Matrix operations
  ✅ Parallel reductions

EXAMPLE:
  matmul on 10,000 elements:
    - CPU: 20 ms
    - GPU: 2 ms (10x faster!)
  
  add on 100 elements:
    - CPU: 0.01 ms
    - GPU: 0.50 ms (slower due to data transfer overhead!)
""")


# ============================================================================
# REAL-WORLD EXAMPLE: Finding a Bug
# ============================================================================
print("\n" + "=" * 70)
print("REAL-WORLD EXAMPLE: Profiling Found a Bug!")
print("=" * 70)

cp.enable_profiling()

# Simulate a buggy implementation with unnecessary work
data = cp.tensor([float(i) for i in range(500)])

# BUG: Accidentally computing mean inside a loop instead of once!
results = []
for i in range(50):
    m = data.mean()  # ← This should be OUTSIDE the loop!
    results.append(data - m)

print("📊 Report showing the bug:")
print(cp.profile_report())

print("""
💡 INSIGHT:
The 'mean' operation was called 50 times!
Looking at the code, we see it's inside a loop but the result doesn't change.

BEFORE (buggy):
  for i in range(50):
      m = data.mean()  # ← Called 50 times
      results.append(data - m)

AFTER (fixed):
  m = data.mean()      # ← Called once!
  for i in range(50):
      results.append(data - m)

Profile-driven development FTW! 🎉
""")

cp.disable_profiling()


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("🎓 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. CONTEXT MATTERS
   - 5 ms on 1M elements is great!
   - 5 ms on 10 elements is terrible!

2. FOCUS ON % TOTAL
   - Optimize operations taking >50% of time first
   - Don't waste time on operations <10% of total

3. CREATE BASELINES
   - Profile typical workloads on your hardware
   - Compare future runs to detect regressions

4. CHECK BACKENDS
   - Large data + matmul → Use GPU
   - Small data + simple ops → Use CPU

5. LOOK FOR PATTERNS
   - High operation count → Maybe batch?
   - Unexpected slow operations → Might be a bug!

GOLDEN RULE:
"Measure, don't guess. Profile, then optimize."
""")


# ============================================================================
# WHAT'S NEXT?
# ============================================================================
print("=" * 70)
print("📚 NEXT STEPS")
print("=" * 70)
print("""
Congratulations! You've completed the Profiling Basics tutorial! 🎉

You now know:
  ✅ How to enable/disable profiling
  ✅ How to generate and read reports
  ✅ How to interpret metrics and find bottlenecks

NEXT:
  👉 Tutorial 02: Intermediate Usage
     - Profile specific code sections (context managers)
     - Profile your own functions (decorators)
     - Get automatic optimization recommendations

  cd ../02_intermediate
  python 01_context_manager.py
""")
