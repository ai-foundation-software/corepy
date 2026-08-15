"""
Tutorial 01-2: Your First Performance Report

WHAT: Generate and understand performance reports
WHY: Reports show you where your code spends time
HOW: Call cp.profile_report() after running profiled code

Expected Time: 3 minutes
"""

import corepy as cp

# ============================================================================
# SCENARIO: Analyzing a Data Processing Pipeline
# ============================================================================
# You're processing some numerical data and want to know which operations
# are taking the most time. Let's profile it!

print("=" * 70)
print("SCENARIO: Profiling a Data Processing Pipeline")
print("=" * 70)
print()


# ============================================================================
# STEP 1: Enable Profiling & Run Your Code
# ============================================================================
print("STEP 1: Running code with profiling enabled...")
print("-" * 70)

cp.enable_profiling()

# Simulate a real data pipeline
raw_data = cp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])

print(f"Raw data: {raw_data}")

# Step 1: Normalize the data (subtract mean)
mean_val = raw_data.mean()
normalized = raw_data - mean_val
print(f"Normalized: {normalized}")

# Step 2: Scale by 2
scaled = normalized * 2.0
print(f"Scaled: {scaled}")

# Step 3: Apply a transformation
transformed = scaled + 100.0
print(f"Transformed: {transformed}")

# Step 4: Compute final statistics
final_sum = transformed.sum()
final_mean = transformed.mean()
print(f"Final sum: {final_sum}, mean: {final_mean}")

print()


# ============================================================================
# STEP 2: Generate the Performance Report
# ============================================================================
print("=" * 70)
print("STEP 2: Performance Report")
print("=" * 70)

# This is the magic line! It shows you everything that happened.
report = cp.profile_report()
print(report)


# ============================================================================
# STEP 3: Understanding the Report
# ============================================================================
print("\n" + "=" * 70)
print("🎓 UNDERSTANDING THE REPORT")
print("=" * 70)
print("""
The report shows:

┌─────────────┬─────────┬─────────┬──────────┬──────────┐
│ Operation   │ Count   │ Avg (ms)│ Backend  │ % Total  │
├─────────────┼─────────┼─────────┼──────────┼──────────┤
│ mean        │ 2       │ 0.15    │ CPU      │ 15%      │ ← Called twice
│ sub         │ 1       │ 0.10    │ CPU      │ 10%      │
│ mul         │ 1       │ 0.12    │ CPU      │ 12%      │
│ add         │ 1       │ 0.08    │ CPU      │  8%      │
│ sum         │ 1       │ 0.55    │ CPU      │ 55%      │ ← Slowest!
└─────────────┴─────────┴─────────┴──────────┴──────────┘

📊 COLUMN MEANINGS:

1. Operation:  The name of the operation (add, mul, mean, etc.)
2. Count:      How many times this operation was called
3. Avg (ms):   Average execution time in milliseconds
4. Backend:    Which hardware executed it (CPU, GPU, etc.)
5. % Total:    Percentage of total execution time

🔍 WHAT TO LOOK FOR:

✅ HIGH % TOTAL = Optimization opportunity!
   In this example, 'sum' takes 55% of time - that's your bottleneck!

✅ HIGH COUNT = Maybe batch these operations?
   If you see an operation called 1000 times, consider batching.

✅ BACKEND MISMATCH = Wrong hardware?
   If you have a GPU but see "CPU", you might want to move data.
""")


# ============================================================================
# STEP 4: Export the Report (Optional)
# ============================================================================
print("=" * 70)
print("STEP 4: Exporting Report (Optional)")
print("=" * 70)

# You can save the report to a file for later analysis
output_path = "/tmp/profile_report.json"
cp.export_profile(output_path, format="json")
print(f"✅ Report saved to: {output_path}")
print("   You can open this in any JSON viewer or load it programmatically.\n")


# ============================================================================
# STEP 5: Clear & Re-Profile
# ============================================================================
print("=" * 70)
print("STEP 5: Clearing Profile Data")
print("=" * 70)

# If you want to profile a different section of code, clear the old data
cp.clear_profile()
print("✅ Profile data cleared. Ready to profile a new section!\n")


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("=" * 70)
print("🎓 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. cp.profile_report()               → View the performance report
2. cp.export_profile(path, format)   → Save report to file
3. cp.clear_profile()                → Reset profiling data

READING REPORTS:
- Look for high "% Total" - these are your bottlenecks
- Check "Count" - high counts might benefit from batching
- Verify "Backend" - make sure you're using the right hardware

WORKFLOW:
1. Enable profiling
2. Run your code
3. Generate report
4. Identify bottlenecks
5. Optimize
6. Re-profile to verify improvements!
""")


# ============================================================================
# WHAT'S NEXT?
# ============================================================================
print("=" * 70)
print("📚 NEXT TUTORIAL")
print("=" * 70)
print("""
Now you can generate reports, but what do the numbers really mean?

👉 Run: python 03_understanding_metrics.py

You'll learn:
- How to interpret timing data
- When to worry about performance
- How to set performance baselines
""")
