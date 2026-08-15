"""
Tutorial 02-1: Context Manager Profiling

WHAT: Profile specific code sections instead of everything
WHY: In real apps, you only care about certain critical sections
HOW: Use ProfileContext to isolate profiling to specific blocks

Expected Time: 5 minutes
"""

import corepy as cp
from corepy.profiler import ProfileContext

# ============================================================================
# PROBLEM: Global Profiling Tracks Too Much
# ============================================================================
print("=" * 70)
print("❌ PROBLEM: Global Profiling Tracks Everything")
print("=" * 70)
print("""
When you use cp.enable_profiling(), it tracks ALL operations:
  - Initialization code (don't care)
  - Data loading (don't care)
  - Critical algorithm (DO care!)
  - Cleanup code (don't care)

This makes reports noisy and hard to analyze.
""")


# ============================================================================
# SOLUTION: Use Context Managers
# ============================================================================
print("=" * 70)
print("✅ SOLUTION: Profile Only What Matters")
print("=" * 70)

# This code runs but is NOT profiled
print("\n1. Initialization (not profiled)...")
data = cp.array([float(i) for i in range(1000)])
print(f"   Created array with {len(data)} elements")

# Enable profiling ONLY for this critical section
print("\n2. Critical algorithm (PROFILED)...")
with ProfileContext("critical_section"):
    # Everything inside this block is tracked!
    normalized = data - data.mean()
    scaled = normalized * 2.0
    result = scaled + 100.0
    final = result.sum()

print(f"   Result: {final}")

# This code runs but is NOT profiled
print("\n3. Cleanup (not profiled)...")
print("   Saving results to disk...")  # (simulated)


# ============================================================================
# View the Targeted Report
# ============================================================================
print("\n" + "=" * 70)
print("📊 PROFILING REPORT (Only 'critical_section')")
print("=" * 70)

# Get report ONLY for the profiled section
report = cp.profile_report(context="critical_section")
print(report)

print("""
💡 NOTICE:
Only operations inside the 'with ProfileContext()' block appear!
This makes the report clean and focused.
""")


# ============================================================================
# EXAMPLE 2: Multiple Independent Sections
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Profiling Multiple Independent Sections")
print("=" * 70)

cp.clear_profile()

# Profile preprocessing
with ProfileContext("preprocessing"):
    data = cp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    processed = (data - data.mean()) / data.std()

# Profile training (separate context!)
with ProfileContext("training"):
    for epoch in range(10):
        loss = processed.sum()  # Simplified training step

# Profile evaluation (another separate context!)
with ProfileContext("evaluation"):
    metrics = processed.mean()

# Now you can get separate reports for each section!
print("\n📊 PREPROCESSING METRICS:")
print(cp.profile_report(context="preprocessing"))

print("\n📊 TRAINING METRICS:")
print(cp.profile_report(context="training"))

print("\n📊 EVALUATION METRICS:")
print(cp.profile_report(context="evaluation"))


# ============================================================================
# EXAMPLE 3: Nested Contexts (Advanced)
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Nested Profiling Contexts")
print("=" * 70)

cp.clear_profile()

# You can nest contexts to create hierarchical reports!
with ProfileContext("outer_pipeline"):
    data = cp.array([float(i) for i in range(100)])

    with ProfileContext("preprocessing"):
        normalized = data - data.mean()

    with ProfileContext("computation"):
        result = normalized * 2.0

    with ProfileContext("aggregation"):
        final = result.sum()

print("\n📊 FULL PIPELINE:")
print(cp.profile_report(context="outer_pipeline"))

print("\n📊 JUST PREPROCESSING:")
print(cp.profile_report(context="preprocessing"))

print("""
💡 PRO TIP:
Use nested contexts to create hierarchical performance reports.
This is great for complex pipelines with multiple stages!
""")


# ============================================================================
# EXAMPLE 4: Comparing Different Implementations
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Compare Different Implementations")
print("=" * 70)

cp.clear_profile()

data = cp.array([float(i) for i in range(1000)])

# Implementation A: Separate operations
with ProfileContext("implementation_A"):
    temp1 = data * 2.0
    temp2 = temp1 + 5.0
    temp3 = temp2 - 1.0
    result_a = temp3.mean()

# Implementation B: Combined operation
with ProfileContext("implementation_B"):
    result_b = ((data * 2.0) + 5.0 - 1.0).mean()

print("\n📊 IMPLEMENTATION A (Separate):")
report_a = cp.profile_report(context="implementation_A")
print(report_a)

print("\n📊 IMPLEMENTATION B (Combined):")
report_b = cp.profile_report(context="implementation_B")
print(report_b)

print("""
💡 USE CASE:
This technique is perfect for A/B testing implementations!
You can objectively measure which approach is faster.

RESULT:
Implementation B might be faster due to:
  - Fewer intermediate allocations
  - Better compiler optimizations
  - Reduced memory traffic
""")


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("🎓 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. USE CONTEXT MANAGERS:
   with ProfileContext("section_name"):
       # Only this code is profiled

2. BENEFIT: FOCUSED REPORTS
   - No noise from initialization/cleanup
   - Easy to interpret
   - Compare different sections

3. GET CONTEXT-SPECIFIC REPORTS:
   cp.profile_report(context="section_name")

4. NEST CONTEXTS:
   Create hierarchical performance reports for complex pipelines

5. A/B TESTING:
   Profile different implementations to objectively compare

WHEN TO USE:
  ✅ Large applications (only profile critical sections)
  ✅ A/B testing implementations
  ✅ Debugging specific slow sections
  ✅ Production monitoring (profile key operations)
""")


# ============================================================================
# WHAT'S NEXT?
# ============================================================================
print("=" * 70)
print("📚 NEXT TUTORIAL")
print("=" * 70)
print("""
Now you can profile specific sections. But what about profiling your
own custom functions?

👉 Run: python 02_custom_decorators.py

You'll learn:
- How to use the @profile_operation decorator
- Profile any Python function
- Integrate profiling into existing code with zero changes
""")
