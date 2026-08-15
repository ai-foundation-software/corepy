"""
Tutorial 03-2: Export & Integration with External Tools

WHAT: Export profiling data to industry-standard formats
WHY: Integrate with existing tools and share with teams
HOW: Multiple export formats for different use cases

Expected Time: 5 minutes
"""

import csv
import json

import corepy as cp
from corepy.profiler import profile_operation

# ============================================================================
# WHY EXPORT? Integration is Key
# ============================================================================
print("=" * 70)
print("📤 WHY EXPORT PROFILING DATA?")
print("=" * 70)
print("""
REASONS TO EXPORT:

1. TOOL INTEGRATION:
   - Use your favorite profiling tools (Speedscope, Chrome DevTools)
   - Integrate with CI/CD pipelines
   - Feed into monitoring systems (Datadog, New Relic)

2. TEAM COLLABORATION:
   - Share performance data with teammates
   - Attach to bug reports / PRs
   - Document optimization efforts

3. LONG-TERM STORAGE:
   - Track performance over time
   - Compare releases
   - Build performance history

4. CUSTOM ANALYSIS:
   - Import into spreadsheets
   - Custom visualizations
   - Statistical analysis
""")


# ============================================================================
# FORMAT 1: JSON (General Purpose)
# ============================================================================
print("=" * 70)
print("FORMAT 1: JSON Export")
print("=" * 70)


@profile_operation
def sample_workflow():
    data = cp.array([float(i) for i in range(1000)])
    normalized = (data - data.mean()) / data.std()
    result = normalized.sum()
    return result


cp.enable_profiling()
result = sample_workflow()

# Export as JSON
json_path = "/tmp/corepy_profile.json"
cp.export_profile(json_path, format="json")

print(f"✅ Exported to: {json_path}")

# Load and inspect
with open(json_path) as f:
    profile_data = json.load(f)

print("\n📋 JSON STRUCTURE:")
print(f"   Total operations: {len(profile_data['operations'])}")
print(f"   Total time: {profile_data['total_time_ms']:.2f}ms")
print(f"   Profiling session: {profile_data['metadata']['session_id']}")

print("\n   First operation:")
first_op = list(profile_data["operations"].values())[0]
print(f"     {json.dumps(first_op, indent=6)}")

print("""
💡 JSON FORMAT USE CASES:

✅ GOOD FOR:
   - Loading into custom analysis tools
   - Programmatic processing (Python, JavaScript, etc.)
   - REST API integration
   - NoSQL databases (MongoDB, etc.)

📊 DATA STRUCTURE:
   {
     "metadata": {...},        // Session info, timestamp, version
     "operations": {...},      // All profiled operations
     "total_time_ms": 123.45,  // Total execution time
     "recommendations": [...]  // Optimization suggestions
   }

EXAMPLE USAGE:
   # Load in Python
   import json
   with open('profile.json') as f:
       data = json.load(f)
   
   # Analyze programmatically
   slow_ops = [op for op in data['operations']
               if op['avg_time_ms'] > 10.0]
""")

cp.clear_profile()


# ============================================================================
# FORMAT 2: CSV (Spreadsheet Analysis)
# ============================================================================
print("\n" + "=" * 70)
print("FORMAT 2: CSV Export")
print("=" * 70)

cp.enable_profiling()

# Run more operations for better CSV example
data = cp.array([float(i) for i in range(500)])
for i in range(5):
    temp = data + i
    result = temp.mean()

# Export as CSV
csv_path = "/tmp/corepy_profile.csv"
cp.export_profile(csv_path, format="csv")

print(f"✅ Exported to: {csv_path}")

# Read and display
print("\n📋 CSV PREVIEW:")
with open(csv_path) as f:
    reader = csv.DictReader(f)
    print(f"   Columns: {', '.join(reader.fieldnames)}\n")
    for i, row in enumerate(reader):
        if i < 3:  # Show first 3 rows
            print(
                f"   {row['operation']:<15} | {row['count']:>5} calls | {row['avg_time_ms']:>8} ms | {row['backend']}"
            )

print("""
💡 CSV FORMAT USE CASES:

✅ GOOD FOR:
   - Excel / Google Sheets analysis
   - Quick data inspection
   - Pivot tables and charts
   - Non-technical stakeholders
   - Reporting and dashboards

📊 CSV COLUMNS:
   - operation: Operation name
   - count: Number of calls
   - avg_time_ms: Average execution time
   - total_time_ms: Total time spent
   - min_time_ms: Fastest call
   - max_time_ms: Slowest call
   - backend: Execution backend (CPU/GPU)
   - percent_total: % of total execution time

EXAMPLE ANALYSIS IN EXCEL:
   1. Open CSV in Excel
   2. Create pivot table
   3. Group by operation
   4. Sort by total_time_ms descending
   5. Create chart showing top 10 slowest operations
""")

cp.clear_profile()


# ============================================================================
# FORMAT 3: Flamegraph (Speedscope)
# ============================================================================
print("\n" + "=" * 70)
print("FORMAT 3: Flamegraph (Speedscope Format)")
print("=" * 70)

# Already covered in detail in 01_flamegraph_analysis.py
# This is a quick reference

flamegraph_path = "/tmp/corepy_flamegraph.json"
cp.enable_profiling()

result = sample_workflow()

cp.export_profile(flamegraph_path, format="flamegraph")

print(f"✅ Exported to: {flamegraph_path}")
print("\n💡 TO VIEW:")
print("   1. Open https://speedscope.app")
print(f"   2. Drag & drop {flamegraph_path}")
print("   3. Explore interactive flamegraph!")

print("""
💡 FLAMEGRAPH USE CASES:

✅ GOOD FOR:
   - Visual performance analysis
   - Understanding call hierarchies
   - Presenting to teams / management
   - Quick bottleneck identification

🔥 VIEWER: Speedscope (https://speedscope.app)
   - Web-based (no installation)
   - Interactive exploration
   - Multiple view modes
   - Search and filtering

See tutorial 01_flamegraph_analysis.py for detailed guide!
""")

cp.clear_profile()


# ============================================================================
# FORMAT 4: Chrome Tracing
# ============================================================================
print("\n" + "=" * 70)
print("FORMAT 4: Chrome Tracing Format")
print("=" * 70)

chrome_path = "/tmp/corepy_trace.json"

cp.enable_profiling()

# Create a timeline of events
for i in range(10):
    data = cp.array([float(j) for j in range(100)])
    result = (data + i).mean()

cp.export_profile(chrome_path, format="chrome_tracing")

print(f"✅ Exported to: {chrome_path}")
print("\n💡 TO VIEW:")
print("   1. Open Chrome browser")
print("   2. Navigate to: chrome://tracing")
print(f"   3. Click 'Load' and select {chrome_path}")
print("   4. Explore timeline view!")

print("""
💡 CHROME TRACING USE CASES:

✅ GOOD FOR:
   - Timeline visualization
   - Understanding execution order
   - Analyzing concurrency
   - Debugging race conditions

📊 CHROME TRACING VIEWER:
   - Timeline view (operations over time)
   - Zooming and panning
   - Per-thread visualization
   - Duration measurement

WHEN TO USE:
   - Multi-threaded code
   - Async operations
   - Timing-sensitive code
   - Parallel execution analysis
""")

cp.clear_profile()


# ============================================================================
# COMPARING FORMATS: Which to Use When?
# ============================================================================
print("\n" + "=" * 70)
print("📊 EXPORT FORMAT COMPARISON")
print("=" * 70)

comparison = """
┌─────────────────┬──────────────┬─────────────────┬─────────────────┐
│ Format          │ Best For     │ Viewer          │ File Size       │
├─────────────────┼──────────────┼─────────────────┼─────────────────┤
│ JSON            │ Programming  │ Any JSON tool   │ Medium          │
│                 │ Integration  │ Python, JS      │                 │
├─────────────────┼──────────────┼─────────────────┼─────────────────┤
│ CSV             │ Spreadsheets │ Excel           │ Small           │
│                 │ Quick Look   │ Google Sheets   │                 │
├─────────────────┼──────────────┼─────────────────┼─────────────────┤
│ Flamegraph      │ Visual       │ Speedscope      │ Medium          │
│                 │ Analysis     │ (speedscope.app)│                 │
├─────────────────┼──────────────┼─────────────────┼─────────────────┤
│ Chrome Tracing  │ Timeline     │ Chrome          │ Large           │
│                 │ Multi-thread │ (chrome://trace)│                 │
└─────────────────┴──────────────┴─────────────────┴─────────────────┘

DECISION TREE:

Need to share with non-technical person?
  → CSV (open in Excel)

Need to debug why code is slow?
  → Flamegraph (visual bottlenecks)

Need to see what happens when?
  → Chrome Tracing (timeline)

Need to integrate with another tool?
  → JSON (most flexible)

Need the smallest file?
  → CSV (most compact)
"""

print(comparison)


# ============================================================================
# PRACTICAL WORKFLOW: Multi-Format Export
# ============================================================================
print("\n" + "=" * 70)
print("🔧 PRACTICAL WORKFLOW")
print("=" * 70)

print("""
RECOMMENDED APPROACH: Export to multiple formats!

# Profile your code
cp.enable_profiling()
run_your_code()

# Export to multiple formats for different uses
cp.export_profile("profile.json", format="json")          # For CI/CD
cp.export_profile("profile.csv", format="csv")            # For Excel
cp.export_profile("profile_flame.json", format="flamegraph")  # For analysis
cp.export_profile("profile_trace.json", format="chrome_tracing")  # For timeline

USAGE:
1. CSV → Quick look in Excel, share with manager
2. Flamegraph → Deep dive analysis in Speedscope
3. JSON → Store in database, track over time
4. Chrome Tracing → Debug specific timing issues

AUTOMATION EXAMPLE (CI/CD):

    # .github/workflows/performance.yml
    - name: Run Performance Tests
      run: |
        python run_benchmark.py
        cp export profile.json
    
    - name: Upload Artifacts
      uses: actions/upload-artifact@v2
      with:
        name: performance-profile
        path: profile.json
    
    - name: Check for Regressions
      run: |
        python check_performance.py profile.json baseline.json
        # Fails build if performance regression detected
""")


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("🎓 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. FOUR EXPORT FORMATS:
   - JSON: Programming, integration
   - CSV: Spreadsheets, quick analysis
   - Flamegraph: Visual analysis (Speedscope)
   - Chrome Tracing: Timeline view

2. EXPORT SYNTAX:
   cp.export_profile(path, format="json"|"csv"|"flamegraph"|"chrome_tracing")

3. MULTI-FORMAT EXPORT:
   Export to multiple formats for different audiences!

4. VIEWERS:
   - JSON: Any text editor, Python, etc.
   - CSV: Excel, Google Sheets
   - Flamegraph: https://speedscope.app
   - Chrome Tracing: chrome://tracing

5. USE CASES:
   - Development: Flamegraph (find bottlenecks)
   - Sharing: CSV (non-technical stakeholders)
   - Storage: JSON (databases, version control)
   - Debugging: Chrome Tracing (timing issues)

WORKFLOW:
  1. Profile your code
  2. Export to appropriate format(s)
  3. Analyze in the right tool
  4. Share findings with team
  5. Track performance over time
""")


# ============================================================================
# WHAT'S NEXT?
# ============================================================================
print("=" * 70)
print("📚 NEXT TUTORIAL")
print("=" * 70)
print("""
You can now export profiling data to any format!
But how do you track performance over time?

👉 Run: python 03_custom_baselines.py

You'll learn:
- Creating performance baselines
- Detecting regressions in CI/CD
- Setting performance budgets
- Automated performance testing
""")
