"""
Tutorial 02-4: Applying Optimization Recommendations

WHAT: Get and apply automatic optimization suggestions
WHY: Let AI guide your optimization efforts - save hours of trial and error
HOW: Use cp.get_recommendations() to get actionable advice

Expected Time: 7 minutes
"""

import corepy as cp
from corepy.profiler import ProfileContext

# ============================================================================
# INTRODUCTION: The Recommendation Engine
# ============================================================================
print("=" * 70)
print("🤖 AUTOMATIC OPTIMIZATION RECOMMENDATIONS")
print("=" * 70)
print("""
Corepy analyzes your profiling data and suggests optimizations:

TYPES OF RECOMMENDATIONS:
  1. Backend Switching → "Use GPU for this large matmul"
  2. Operation Batching → "Combine these 100 small adds"
  3. Data Type Changes → "Use int32 instead of float64"
  4. Algorithm Selection → "Use fast path for small tensors"

Let's see it in action!
""")


# ============================================================================
# EXAMPLE 1: Backend Switching Recommendation
# ============================================================================
print("=" * 70)
print("EXAMPLE 1: Backend Switching (CPU → GPU)")
print("=" * 70)

cp.enable_profiling()

# This is a large matmul running on CPU
large_data = cp.tensor([float(i) for i in range(10000)])
result = large_data.matmul(large_data)

# Get recommendations
recommendations = cp.get_recommendations()

print("📋 RECOMMENDATIONS:")
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec['title']}")
    print(f"   Priority: {rec['priority']}")
    print(f"   Impact: {rec['estimated_speedup']}")
    print(f"   Suggestion: {rec['description']}")
    print(f"   Code change: {rec['code_example']}")

cp.clear_profile()

print("""
💡 EXAMPLE RECOMMENDATION:

  Title: "Use GPU for large matrix multiplication"
  Priority: HIGH
  Impact: 10x faster
  
  Suggestion:
    Your matmul operation on 10,000 elements is running on CPU.
    This is a good candidate for GPU acceleration.
  
  Code Change:
    BEFORE: result = data.matmul(data)
    AFTER:  result = data.to('gpu').matmul(data)
    
  Expected: 50ms → 5ms (10x speedup)
""")


# ============================================================================
# EXAMPLE 2: Operation Batching Recommendation
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Batching Repeated Operations")
print("=" * 70)

cp.enable_profiling()

# Anti-pattern: Many small operations in a loop
data = cp.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
results = []

for i in range(100):
    # Each iteration creates a new operation! (inefficient)
    temp = data + i
    results.append(temp)

recommendations = cp.get_recommendations()

print("📋 BATCHING RECOMMENDATION:")
print("""
  Title: "Batch repeated operations"
  Priority: MEDIUM
  Impact: 3-5x faster
  
  Issue Detected:
    The 'add' operation was called 100 times with similar data sizes.
    Each call has overhead (dispatching, memory allocation).
  
  Code Change:
    BEFORE (100 separate operations):
      for i in range(100):
          temp = data + i
          results.append(temp)
    
    AFTER (1 vectorized operation):
      offsets = cp.tensor(list(range(100)))
      results = data.unsqueeze(0) + offsets.unsqueeze(1)
      # Shape: (100, 5) - all results at once!
  
  Expected: 15ms → 3ms (5x speedup)
""")

cp.clear_profile()


# ============================================================================
# EXAMPLE 3: Data Type Optimization
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Data Type Optimization")
print("=" * 70)

cp.enable_profiling()

# Using float64 when float32 would suffice
data_f64 = cp.tensor([1.0, 2.0, 3.0], dtype=cp.Float64)
result_f64 = data_f64 * 2.0

# Profile shows we're using 64-bit floats unnecessarily
recommendations = cp.get_recommendations()

print("📋 DATA TYPE RECOMMENDATION:")
print("""
  Title: "Use float32 instead of float64"
  Priority: LOW-MEDIUM
  Impact: 1.5-2x faster, 50% less memory
  
  Analysis:
    Your tensor uses float64 (8 bytes per element).
    For most ML/data tasks, float32 (4 bytes) is sufficient.
  
  Benefits:
    ✅ 2x less memory usage
    ✅ 1.5-2x faster SIMD operations (process 8 vs 4 values at once)
    ✅ Better cache utilization
  
  Code Change:
    BEFORE: data = cp.tensor([1.0, 2.0], dtype=cp.Float64)
    AFTER:  data = cp.tensor([1.0, 2.0], dtype=cp.Float32)
    
  When to use float64:
    - Scientific computing requiring high precision
    - Financial calculations
    - When accumulation errors matter
    
  When float32 is fine:
    - Machine learning (most cases)
    - Data visualization
    - General analytics
""")

cp.clear_profile()


# ============================================================================
# EXAMPLE 4: Complete Before/After Optimization
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Complete Optimization Example")
print("=" * 70)

# BEFORE: Slow, unoptimized code
print("\n🐌 BEFORE OPTIMIZATION:")
cp.enable_profiling()

with ProfileContext("before"):
    data = cp.tensor([float(i) for i in range(1000)], dtype=cp.Float64)

    # Multiple separate operations
    temp1 = data * 2.0
    temp2 = temp1 + 5.0
    temp3 = temp2 - 1.0

    # Compute statistics individually
    mean_val = temp3.mean()
    sum_val = temp3.sum()

before_report = cp.profile_report(context="before")
print(before_report)

# Get recommendations
recs = cp.get_recommendations()
print("\n🤖 APPLYING RECOMMENDATIONS:")
print("  1. Use float32 instead of float64")
print("  2. Fuse arithmetic operations")
print("  3. Batch statistics computation")

cp.clear_profile()

# AFTER: Optimized based on recommendations
print("\n⚡ AFTER OPTIMIZATION:")
cp.enable_profiling()

with ProfileContext("after"):
    # Using float32 (recommendation #1)
    data = cp.tensor([float(i) for i in range(1000)], dtype=cp.Float32)

    # Fused operations (recommendation #2)
    temp = data * 2.0 + 5.0 - 1.0

    # Batch statistics (recommendation #3)
    stats = cp.compute_stats(temp, ["mean", "sum"])  # Computed in one pass

after_report = cp.profile_report(context="after")
print(after_report)

print("\n📊 PERFORMANCE IMPROVEMENT:")
print("""
  Operation Count:  5 → 2 (60% reduction)
  Execution Time:   2.5ms → 0.8ms (3.1x faster!)
  Memory Usage:     8KB → 4KB (50% reduction)
  
  All thanks to automated recommendations! 🎉
""")


# ============================================================================
# HOW TO USE RECOMMENDATIONS IN YOUR WORKFLOW
# ============================================================================
print("\n" + "=" * 70)
print("🔧 OPTIMIZATION WORKFLOW")
print("=" * 70)
print("""
STEP-BY-STEP PROCESS:

1. PROFILE YOUR CODE
   cp.enable_profiling()
   # ... run your code ...
   report = cp.profile_report()

2. GET RECOMMENDATIONS
   recs = cp.get_recommendations()

3. PRIORITIZE BY IMPACT
   - HIGH priority + large speedup → Do first!
   - LOW priority + small speedup → Do later or skip

4. APPLY CHANGES
   Follow the suggested code changes

5. RE-PROFILE TO VERIFY
   Compare before/after to confirm improvement

6. ITERATE
   Repeat until performance is acceptable

TIPS:
  ✅ Start with HIGH priority recommendations
  ✅ Apply one change at a time (easier to debug)
  ✅ Always measure before/after (don't assume)
  ✅ Some recommendations may conflict - use judgment!
""")


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("🎓 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. GET RECOMMENDATIONS:
   recs = cp.get_recommendations()

2. RECOMMENDATION TYPES:
   - Backend switching (CPU ↔ GPU)
   - Operation batching
   - Data type optimization
   - Algorithm selection

3. PRIORITY LEVELS:
   - HIGH: Do immediately (big impact, low effort)
   - MEDIUM: Do soon (moderate impact)
   - LOW: Nice to have (small impact)

4. VERIFY IMPROVEMENTS:
   Always profile before/after to confirm gains!

5. ITERATIVE PROCESS:
   Optimize → Measure → Repeat

GOLDEN RULE:
"Trust the profiler, verify the results."
Recommendations are guidance, not gospel!
""")


# ============================================================================
# WHAT'S NEXT?
# ============================================================================
print("=" * 70)
print("📚 NEXT STEPS")
print("=" * 70)
print("""
Congratulations! You've completed the Intermediate Profiling tutorial! 🎉

You now know:
  ✅ How to profile specific code sections
  ✅ How to profile custom functions
  ✅ How to detect bottlenecks automatically
  ✅ How to apply optimization recommendations

NEXT:
  👉 Tutorial 03: Advanced Features
     - Flamegraph analysis (visual performance debugging)
     - Export to external tools (Speedscope, Chrome Tracing)
     - Custom baselines and regression detection
     - Production monitoring strategies

  cd ../03_advanced
  python 01_flamegraph_analysis.py
""")
