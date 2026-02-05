"""
Tutorial 02-3: Automatic Bottleneck Detection

WHAT: Let corepy automatically find slow operations
WHY: Manual analysis is time-consuming and error-prone
HOW: Built-in algorithms detect performance anomalies

Expected Time: 5 minutes
"""

import corepy as cp
from corepy.profiler import profile_operation

# ============================================================================
# INTRODUCTION: What is a Bottleneck?
# ============================================================================
print("=" * 70)
print("🔍 WHAT IS A BOTTLENECK?")
print("=" * 70)
print("""
A bottleneck is an operation that:
  1. Takes a disproportionate amount of total time
  2. Limits the overall performance of your code
  3. Has the highest optimization ROI

EXAMPLE:
  Operation A: 0.1ms (5% of time)
  Operation B: 0.2ms (10% of time)
  Operation C: 1.7ms (85% of time) ← BOTTLENECK!

Optimizing C by 50% saves 0.85ms
Optimizing A by 90% saves only 0.09ms

Focus on the bottleneck!
""")


# ============================================================================
# AUTOMATIC DETECTION: Basic Example
# ============================================================================
print("=" * 70)
print("EXAMPLE 1: Automatic Bottleneck Detection")
print("=" * 70)

cp.enable_profiling()

# Simulate a pipeline with one slow operation
data = cp.tensor([float(i) for i in range(10000)])

# Fast operations
temp1 = data + 1.0  # ~0.2ms
temp2 = data * 2.0  # ~0.2ms

# SLOW operation (bottleneck!)
result = data.matmul(data)  # ~50ms (much slower!)

# Fast operation
final = result.sum()  # ~0.3ms

# Get bottleneck analysis
bottlenecks = cp.detect_bottlenecks()

print("\n🔍 DETECTED BOTTLENECKS:")
for b in bottlenecks:
    print(f"\n⚠️  {b['operation']}")
    print(f"   Severity: {b['severity']}")
    print(f"   Time: {b['time_ms']:.2f}ms ({b['percent_total']:.1f}% of total)")
    print(f"   Why: {b['reason']}")
    print(f"   Suggestion: {b['suggestion']}")

print("""
💡 AUTOMATIC DETECTION:
Corepy found that 'matmul' is a bottleneck because:
  - It takes >50% of total execution time
  - It's significantly slower than other operations
  - Optimizing it would have the biggest impact

You didn't have to analyze anything manually!
""")

cp.clear_profile()


# ============================================================================
# EXAMPLE 2: Multiple Bottlenecks
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Multiple Bottleneck Detection")
print("=" * 70)

cp.enable_profiling()

data = cp.tensor([float(i) for i in range(5000)])

# Several operations with varying costs
for i in range(10):
    temp1 = data + i  # Many small operations

# Two potential bottlenecks
stats = data.std()  # Moderately slow (~2ms)
correlation = data.matmul(data)  # Very slow (~20ms)

# More fast operations
result = data.mean()

bottlenecks = cp.detect_bottlenecks(
    threshold=0.15  # Flag operations taking >15% of time
)

print("\n🔍 BOTTLENECK ANALYSIS:")
print(f"Found {len(bottlenecks)} bottlenecks:\n")

for i, b in enumerate(bottlenecks, 1):
    print(f"{i}. {b['operation']} - {b['severity']} severity")
    print(f"   {b['percent_total']:.1f}% of total time")
    print(f"   💡 {b['suggestion']}\n")

print("""
💡 PRIORITIZATION:
Bottlenecks are ranked by:
  1. Severity (CRITICAL > HIGH > MEDIUM > LOW)
  2. Percentage of total time
  3. Optimization potential

Fix the CRITICAL ones first!
""")

cp.clear_profile()


# ============================================================================
# EXAMPLE 3: Bottleneck Detection in Custom Functions
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Bottlenecks in Custom Code")
print("=" * 70)


@profile_operation
def fast_preprocessing(data):
    """Quick normalization"""
    return (data - data.mean()) / data.std()


@profile_operation
def slow_feature_engineering(data):
    """Expensive feature computation"""
    # Simulate complex computation
    result = data.copy()
    for _ in range(50):
        result = result * 1.001 + 0.001
    return result


@profile_operation
def fast_postprocessing(data):
    """Simple transformation"""
    return data + 100.0


cp.enable_profiling()

data = cp.tensor([float(i) for i in range(1000)])

# Run pipeline
preprocessed = fast_preprocessing(data)
features = slow_feature_engineering(preprocessed)
final = fast_postprocessing(features)

# Detect bottlenecks (including custom functions!)
bottlenecks = cp.detect_bottlenecks()

print("\n🔍 CUSTOM FUNCTION BOTTLENECKS:")
for b in bottlenecks:
    print(f"\n⚠️  Function: {b['operation']}")
    print(f"   Why it's slow: {b['reason']}")
    print(f"   Impact: {b['percent_total']:.1f}% of pipeline")
    print(f"   Action: {b['suggestion']}")

print("""
💡 PRACTICAL INSIGHT:
Bottleneck detection works on:
  ✅ Tensor operations (add, matmul, sum, etc.)
  ✅ Custom functions (decorated with @profile_operation)
  ✅ Any profiled code!

This helps you find slow code ANYWHERE in your application.
""")

cp.clear_profile()


# ============================================================================
# EXAMPLE 4: Performance Degradation Over Time
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Detecting Performance Regressions")
print("=" * 70)

# Set a baseline (from previous profiling sessions)
baseline = {
    "preprocess": 1.0,  # ms
    "compute": 5.0,  # ms
    "aggregate": 0.5,  # ms
}


@profile_operation
def preprocess(data):
    return data - data.mean()


@profile_operation
def compute(data):
    return data.matmul(data)


@profile_operation
def aggregate(data):
    return data.sum()


cp.enable_profiling()

data = cp.tensor([float(i) for i in range(1000)])

for _ in range(10):
    p = preprocess(data)
    c = compute(p)
    a = aggregate(c)

# Check for regressions (operations slower than baseline)
regressions = cp.detect_regressions(baseline)

if regressions:
    print("\n⚠️  PERFORMANCE REGRESSIONS DETECTED:")
    for reg in regressions:
        print(f"\n  Function: {reg['operation']}")
        print(f"  Expected: {reg['baseline_ms']:.2f}ms")
        print(f"  Actual: {reg['actual_ms']:.2f}ms")
        print(f"  Slowdown: {reg['slowdown_factor']:.1f}x")
        print(f"  Possible causes: {', '.join(reg['causes'])}")
else:
    print("\n✅ No performance regressions detected!")

print("""
💡 REGRESSION DETECTION:
By comparing current performance to baselines, you can:
  ✅ Detect when new code makes things slower
  ✅ Catch performance bugs in CI/CD
  ✅ Monitor production performance over time

This is essential for maintaining performance at scale!
""")


# ============================================================================
# HOW BOTTLENECK DETECTION WORKS
# ============================================================================
print("\n" + "=" * 70)
print("🔬 HOW IT WORKS INTERNALLY")
print("=" * 70)
print("""
Corepy uses multiple heuristics to detect bottlenecks:

1. PERCENTAGE THRESHOLD
   - Operations taking >20% of total time are flagged
   - Configurable: detect_bottlenecks(threshold=0.15)

2. OUTLIER DETECTION
   - Operations much slower than average are flagged
   - Statistical analysis (mean + 2*stddev)

3. COUNT ANALYSIS
   - High operation count + low per-op time = batching opportunity
   - Example: 1000 calls to 'add' → suggest vectorization

4. BACKEND MISMATCH
   - Large data on CPU when GPU available
   - Suggests backend switch

5. COMPARISON TO BASELINES
   - Detect performance regressions over time
   - Requires baseline data from previous runs

SEVERITY LEVELS:
  🔴 CRITICAL: >50% of time, immediate action needed
  🟠 HIGH: 20-50% of time, optimize soon
  🟡 MEDIUM: 10-20% of time, consider optimizing
  🟢 LOW: <10% of time, not worth optimizing now
""")


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("🎓 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. AUTOMATIC DETECTION:
   bottlenecks = cp.detect_bottlenecks()
   # Returns list of slow operations with suggestions

2. CONFIGURATION:
   cp.detect_bottlenecks(
       threshold=0.20,     # Flag ops taking >20% of time
       min_calls=10        # Ignore ops called <10 times
   )

3. REGRESSION DETECTION:
   regressions = cp.detect_regressions(baseline)
   # Compare current vs. expected performance

4. WORKS EVERYWHERE:
   - Tensor operations
   - Custom functions (@profile_operation)
   - Entire code sections (ProfileContext)

5. ACTIONABLE OUTPUT:
   Each bottleneck includes:
   - What's slow
   - Why it's slow
   - How to fix it

WORKFLOW:
  1. Enable profiling
  2. Run your code
  3. Call cp.detect_bottlenecks()
  4. Fix the CRITICAL/HIGH severity issues
  5. Re-profile to verify improvements
""")


# ============================================================================
# WHAT'S NEXT?
# ============================================================================
print("=" * 70)
print("📚 NEXT TUTORIAL")
print("=" * 70)
print("""
You can now automatically find bottlenecks!
But what should you DO about them?

👉 Run: python 04_optimization_tips.py

You'll learn:
- How to apply optimization recommendations
- Real before/after examples
- Measuring optimization impact
- Common optimization patterns
""")
