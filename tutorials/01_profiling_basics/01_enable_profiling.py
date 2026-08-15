"""
Tutorial 01-1: Enabling Performance Profiling

WHAT: Learn how to turn profiling on/off
WHY: Profiling helps you understand where your code spends time
HOW: Single function call with zero configuration needed

Expected Time: 2 minutes
"""

import corepy as cp

# ============================================================================
# STEP 1: Enable Profiling
# ============================================================================
# This single line enables automatic tracking of ALL array operations.
# There's no configuration needed - it just works!

print("=" * 60)
print("STEP 1: Enabling Profiling")
print("=" * 60)

cp.enable_profiling()
print("✅ Profiling enabled! All operations will now be tracked.\n")

# NOTE: When profiling is enabled, corepy automatically records:
#   - Operation name (e.g., "add", "matmul", "sum")
#   - Execution time (in milliseconds)
#   - Backend used (CPU or GPU)
#   - Data size (number of elements)
#   - Memory usage (coming soon)


# ============================================================================
# STEP 2: Run Some Operations (They'll Be Tracked Automatically)
# ============================================================================
print("=" * 60)
print("STEP 2: Running Operations (Being Profiled)")
print("=" * 60)

# Create a simple array
data = cp.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Created array: {data}")

# Perform an addition (this will be tracked!)
result1 = data + 10.0
print(f"After adding 10: {result1}")

# Perform a multiplication (this will also be tracked!)
result2 = data * 2.0
print(f"After multiplying by 2: {result2}")

# Compute the sum (tracked!)
total = result2.sum()
print(f"Sum of all elements: {total}\n")


# ============================================================================
# STEP 3: Disable Profiling
# ============================================================================
print("=" * 60)
print("STEP 3: Disabling Profiling")
print("=" * 60)

cp.disable_profiling()
print("✅ Profiling disabled. Operations after this won't be tracked.\n")

# Any operations here won't be profiled
data2 = cp.array([6.0, 7.0, 8.0])
result3 = data2 + 1.0  # Not tracked!


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("=" * 60)
print("🎓 KEY TAKEAWAYS")
print("=" * 60)
print("""
1. Enable with:  cp.enable_profiling()
2. Disable with: cp.disable_profiling()
3. All operations between enable/disable are automatically tracked
4. Zero configuration needed!
5. When disabled, there's ZERO performance overhead

PERFORMANCE TIP:
Only enable profiling when you need it. For production code, keep it
disabled unless you're actively debugging performance issues.
""")


# ============================================================================
# WHAT'S NEXT?
# ============================================================================
print("=" * 60)
print("📚 NEXT TUTORIAL")
print("=" * 60)
print("""
Now that you know how to enable profiling, let's see the actual data!

👉 Run: python 02_first_report.py

You'll learn how to generate and interpret performance reports.
""")
