"""
Tutorial 02-2: Profiling Custom Functions with Decorators

WHAT: Use @profile_operation to profile your own functions
WHY: Track performance of custom data processing pipelines
HOW: Simple decorator that wraps any Python function

Expected Time: 5 minutes
"""

import corepy as cp
from corepy.profiler import profile_operation

# ============================================================================
# PROBLEM: How to Profile Your Own Functions?
# ============================================================================
print("=" * 70)
print("🤔 PROBLEM: Profiling Custom Functions")
print("=" * 70)
print("""
You've profiled array operations, but what about YOUR code?
  - Custom data preprocessing functions
  - Feature engineering pipelines
  - Complex business logic

You want to know:
  ✅ How long does my preprocess_data() function take?
  ✅ Is my feature_engineering() a bottleneck?
  ✅ Which of my 5 processing steps is slowest?
""")


# ============================================================================
# SOLUTION: Use the @profile_operation Decorator
# ============================================================================
print("\n" + "=" * 70)
print("✅ SOLUTION: The @profile_operation Decorator")
print("=" * 70)


# Just add @profile_operation above any function!
@profile_operation
def preprocess_data(data):
    """Custom preprocessing function - automatically profiled!"""
    # Normalize
    normalized = (data - data.mean()) / data.std()
    # Scale
    scaled = normalized * 100.0
    return scaled


@profile_operation
def compute_features(data):
    """Feature engineering - automatically profiled!"""
    # Compute various statistics
    features = {
        "mean": data.mean(),
        "sum": data.sum(),
        "max": data.max(),
        "min": data.min(),
    }
    return features


@profile_operation
def apply_transformation(data):
    """Apply final transformation - automatically profiled!"""
    return data + 42.0


# Now run your functions normally
print("\nRunning custom functions...")
cp.enable_profiling()

data = cp.array([float(i) for i in range(1000)])
processed = preprocess_data(data)
features = compute_features(processed)
final = apply_transformation(processed)

# All your custom functions appear in the report!
print("\n📊 PROFILING REPORT (Including Custom Functions):")
print(cp.profile_report())

print("""
💡 NOTICE:
Your custom functions ('preprocess_data', 'compute_features', etc.)
appear alongside corepy array operations!

This lets you answer:
  - Which function is slowest overall?
  - Which function should I optimize first?
  - Are array ops or my custom logic the bottleneck?
""")

cp.clear_profile()


# ============================================================================
# EXAMPLE 2: Comparing Different Implementations
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: A/B Testing with Decorators")
print("=" * 70)


# Two different ways to compute the same thing
@profile_operation
def method_a_separate_ops(data):
    """Compute using separate operations"""
    temp1 = data * 2.0
    temp2 = temp1 + 5.0
    temp3 = temp2 - 1.0
    return temp3.mean()


@profile_operation
def method_b_fused_ops(data):
    """Compute using fused operations"""
    return ((data * 2.0) + 5.0 - 1.0).mean()


cp.enable_profiling()

data = cp.array([float(i) for i in range(1000)])

# Run both methods multiple times
for _ in range(10):
    result_a = method_a_separate_ops(data)
    result_b = method_b_fused_ops(data)

print("\n📊 COMPARISON REPORT:")
report = cp.profile_report()
print(report)

print("""
💡 USE CASE:
You can objectively measure which implementation is faster!

The report shows:
  - method_a_separate_ops: Called 10x, avg X ms
  - method_b_fused_ops: Called 10x, avg Y ms

Choose the faster one based on DATA, not intuition!
""")

cp.clear_profile()


# ============================================================================
# EXAMPLE 3: Nested Function Profiling
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Profiling Function Call Hierarchies")
print("=" * 70)


@profile_operation
def load_and_validate(filepath):
    """Simulated data loading"""
    # In real code: data = cp.read_csv(filepath)
    data = cp.array([float(i) for i in range(500)])
    # Validation
    if data.sum() < 0:
        raise ValueError("Invalid data")
    return data


@profile_operation
def engineer_features(raw_data):
    """Feature engineering pipeline"""
    # This calls other profiled functions!
    processed = preprocess_data(raw_data)
    features = compute_features(processed)
    return features


@profile_operation
def full_pipeline(filepath):
    """Complete data pipeline - calls multiple profiled functions"""
    raw = load_and_validate(filepath)
    features = engineer_features(raw)
    return features


cp.enable_profiling()

result = full_pipeline("data.csv")

print("\n📊 HIERARCHICAL REPORT:")
print(cp.profile_report())

print("""
💡 NESTED PROFILING:
When profiled functions call other profiled functions, you see:
  - full_pipeline: Total time including all sub-calls
  - engineer_features: Time spent in this function
  - load_and_validate: Time spent in this function
  - preprocess_data: Time spent in this function
  - compute_features: Time spent in this function

This helps you understand:
  ✅ Which top-level function is slow?
  ✅ Which sub-function is the bottleneck?
  ✅ Where should optimization efforts focus?
""")

cp.clear_profile()


# ============================================================================
# EXAMPLE 4: Decorator with Context Manager (Combined)
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Combining Decorators + Context Managers")
print("=" * 70)

from corepy.profiler import ProfileContext


@profile_operation
def training_step(data, epoch):
    """Single training iteration"""
    # Forward pass
    prediction = data * 2.0 + 1.0
    # Loss computation
    loss = (prediction - data).sum()
    return loss


cp.enable_profiling()

# Profile just the training loop, not setup
data = cp.array([float(i) for i in range(100)])  # Not profiled

with ProfileContext("training"):
    # Only this loop is profiled
    for epoch in range(10):
        loss = training_step(data, epoch)

# Get report only for training
print("\n📊 TRAINING LOOP ONLY:")
print(cp.profile_report(context="training"))

print("""
💡 BEST OF BOTH WORLDS:
  - Context manager: Profile specific code sections
  - Decorator: Track individual function calls

Together they give you:
  ✅ Focused profiling (only training loop)
  ✅ Function-level granularity (training_step details)
""")


# ============================================================================
# PRACTICAL TIPS
# ============================================================================
print("\n" + "=" * 70)
print("💡 PRACTICAL TIPS")
print("=" * 70)
print("""
1. WHEN TO USE DECORATORS:
   ✅ Function is called multiple times (see average time)
   ✅ Comparing different implementations
   ✅ Tracking performance over time
   ✅ Production monitoring of key functions

2. WHEN TO USE CONTEXT MANAGERS:
   ✅ One-time code blocks (setup, initialization)
   ✅ Profiling specific sections in large apps
   ✅ A/B testing code paths

3. DECORATOR OVERHEAD:
   - Very minimal (<0.1ms per call)
   - Safe to leave in production code
   - Can disable globally with cp.disable_profiling()

4. DON'T OVER-DECORATE:
   ❌ Don't decorate functions called millions of times
   ❌ Don't decorate trivial functions (getters, setters)
   ✅ DO decorate functions that do meaningful work
""")


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("🎓 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. DECORATOR SYNTAX:
   @profile_operation
   def my_function(data):
       # Automatically profiled!
       ...

2. BENEFITS:
   - Track custom Python functions alongside array ops
   - Compare different implementations objectively
   - See function call hierarchies
   - Production monitoring

3. COMBINE WITH CONTEXT MANAGERS:
   Use both for maximum flexibility!

4. VIEW RESULTS:
   cp.profile_report() shows ALL profiled operations
   (both array ops and custom functions)

WORKFLOW:
  1. Add @profile_operation to key functions
  2. Run your code
  3. Check cp.profile_report()
  4. Optimize the slowest functions
  5. Re-profile to verify improvement
""")


# ============================================================================
# WHAT'S NEXT?
# ============================================================================
print("=" * 70)
print("📚 NEXT TUTORIAL")
print("=" * 70)
print("""
You can now profile both array operations AND custom functions.
But how do you FIND bottlenecks automatically?

👉 Run: python 03_bottleneck_detection.py

You'll learn:
- Automatic bottleneck detection
- Performance thresholds
- Anomaly detection
- When to worry about performance
""")
