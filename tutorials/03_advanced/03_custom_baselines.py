"""
Tutorial 03-3: Custom Baselines & Regression Detection

WHAT: Set performance targets and detect when code gets slower
WHY: Prevent performance regressions before they reach production
HOW: Create baselines and automate comparison in CI/CD

Expected Time: 8 minutes
"""

import json

import corepy as cp
from corepy.profiler import profile_operation

# ============================================================================
# PERFORMANCE REGRESSIONS: The Hidden Problem
# ============================================================================
print("=" * 70)
print("⚠️  THE PERFORMANCE REGRESSION PROBLEM")
print("=" * 70)
print("""
SCENARIO:
  Week 1: Your code runs in 100ms ✅
  Week 2: Someone adds a feature, now 150ms ⚠️
  Week 3: Another optimization, now 200ms ❌
  Week 4: Users complain it's too slow 🔥

PROBLEM:
  - Performance degradation happens gradually
  - No one notices until it's too late
  - Hard to identify which change caused it
  - Expensive to fix (multiple PRs to investigate)

SOLUTION:
  - Set performance baselines (expected performance)
  - Automatically detect regressions in CI/CD
  - Fail builds that make code slower
  - Catch issues before they reach production
""")


# ============================================================================
# STEP 1: Creating a Baseline
# ============================================================================
print("=" * 70)
print("STEP 1: Creating a Performance Baseline")
print("=" * 70)


@profile_operation
def data_processing_pipeline(data_size):
    """Standard data processing workflow"""
    data = cp.array([float(i) for i in range(data_size)])
    normalized = (data - data.mean()) / data.std()
    scaled = normalized * 100.0
    result = scaled.sum()
    return result


# Profile the current "good" version
print("\n📊 Profiling baseline (current good performance)...")
cp.enable_profiling()

# Run multiple times for stable measurements
for _ in range(10):
    result = data_processing_pipeline(1000)

# Get the profile data
baseline_report = cp.profile_report(format="dict")

# Save as baseline
baseline_path = "/tmp/performance_baseline.json"
with open(baseline_path, "w") as f:
    json.dump(baseline_report, f, indent=2)

print(f"✅ Baseline saved to: {baseline_path}")
print("\n📋 BASELINE SUMMARY:")
print(f"   Total operations: {len(baseline_report['operations'])}")
print(f"   Total time: {baseline_report['total_time_ms']:.2f}ms")

for op_name, op_data in baseline_report["operations"].items():
    print(f"   {op_name:15s}: {op_data['avg_time_ms']:6.2f}ms avg")

print("""
💡 BASELINE BEST PRACTICES:

1. RUN MULTIPLE TIMES:
   - Average out noise and variance
   - Usually 10-100 iterations
   - More for micro-benchmarks

2. REPRESENTATIVE DATA:
   - Use typical workload sizes
   - Cover edge cases (small, medium, large)
   - Production-like conditions

3. STABLE ENVIRONMENT:
   - Same hardware
   - No background processes
   - Consistent data

4. VERSION CONTROL:
   - Commit baseline.json to repo
   - Update when intentional changes happen
   - Track baseline history
""")

cp.clear_profile()


# ============================================================================
# STEP 2: Detecting Regressions
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Detecting Performance Regressions")
print("=" * 70)


# Simulate a new version with a performance regression
@profile_operation
def data_processing_pipeline_v2(data_size):
    """New version with accidental regression"""
    data = cp.array([float(i) for i in range(data_size)])

    # BUG: Computing mean twice (should be once!)
    normalized = (data - data.mean()) / data.std()

    # BUG: Unnecessary extra operation
    temp = normalized + data.mean()  # ← Regression! Extra mean() call

    scaled = temp * 100.0
    result = scaled.sum()
    return result


# Profile the new version
print("\n📊 Profiling new version (with regression)...")
cp.enable_profiling()

for _ in range(10):
    result = data_processing_pipeline_v2(1000)

new_report = cp.profile_report(format="dict")

# Load baseline for comparison
with open(baseline_path) as f:
    baseline = json.load(f)

# Detect regressions
print("\n🔍 REGRESSION ANALYSIS:")

regressions = []
improvements = []
THRESHOLD = 1.2  # 20% slower = regression

for op_name, new_data in new_report["operations"].items():
    if op_name in baseline["operations"]:
        baseline_time = baseline["operations"][op_name]["avg_time_ms"]
        new_time = new_data["avg_time_ms"]
        ratio = new_time / baseline_time

        if ratio > THRESHOLD:
            regressions.append(
                {
                    "operation": op_name,
                    "baseline_ms": baseline_time,
                    "new_ms": new_time,
                    "slowdown": ratio,
                    "severity": "HIGH" if ratio > 2.0 else "MEDIUM",
                }
            )
        elif ratio < 0.9:  # 10% faster
            improvements.append({"operation": op_name, "speedup": 1.0 / ratio})

if regressions:
    print("\n❌ PERFORMANCE REGRESSIONS DETECTED:")
    for reg in regressions:
        print(f"\n  [{reg['severity']}] {reg['operation']}")
        print(f"    Baseline: {reg['baseline_ms']:.2f}ms")
        print(f"    Current:  {reg['new_ms']:.2f}ms")
        print(f"    Slowdown: {reg['slowdown']:.1f}x slower")
else:
    print("\n✅ No regressions detected!")

if improvements:
    print("\n🎉 PERFORMANCE IMPROVEMENTS:")
    for imp in improvements:
        print(f"  {imp['operation']}: {imp['speedup']:.1f}x faster")


print("""
💡 REGRESSION DETECTION LOGIC:

THRESHOLDS:
  - >20% slower: Flag as regression
  - >100% slower (2x): HIGH severity
  - <10% faster: Flag as improvement

WHY 20%?
  - Accounts for measurement noise
  - Ignores insignificant changes
  - Focuses on meaningful regressions

SEVERITY LEVELS:
  - HIGH: >2x slower (critical issue)
  - MEDIUM: 1.2-2x slower (needs investigation)
  - LOW: <1.2x slower (within noise)
""")

cp.clear_profile()


# ============================================================================
# STEP 3: CI/CD Integration
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: CI/CD Integration (GitHub Actions Example)")
print("=" * 70)

# Example CI/CD script
ci_script = """
# .github/workflows/performance-check.yml

name: Performance Regression Check

on: [pull_request]

jobs:
  performance:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install corepy pytest
      
      - name: Run performance benchmarks
        run: |
          python benchmarks/run_benchmarks.py
          # This generates current_profile.json
      
      - name: Download baseline
        run: |
          # Get baseline from main branch
          git fetch origin main
          git show origin/main:performance_baseline.json > baseline.json
      
      - name: Check for regressions
        run: |
          python scripts/check_performance_regression.py \\
            --current current_profile.json \\
            --baseline baseline.json \\
            --threshold 1.2 \\
            --fail-on-regression
          # Exit code 1 if regression detected = fail build!
      
      - name: Upload performance report
        if: always()
        uses: actions/upload-artifact@v2
        with:
          name: performance-report
          path: |
            current_profile.json
            performance_report.html
"""

with open("/tmp/github_actions_example.yml", "w") as f:
    f.write(ci_script)

print("✅ Example CI/CD config saved to: /tmp/github_actions_example.yml")

print("""
💡 CI/CD WORKFLOW:

1. TRIGGER: On every pull request

2. RUN BENCHMARKS:
   - Execute performance test suite
   - Profile all critical operations
   - Export to current_profile.json

3. COMPARE TO BASELINE:
   - Load baseline from main branch
   - Run regression detection script
   - Flag any operation >20% slower

4. FAIL BUILD IF REGRESSION:
   - exit code 1 = regression detected
   - Prevents merge until fixed
   - Requires performance justification

5. REPORT:
   - Upload profile artifacts
   - Comment on PR with results
   - Link to detailed analysis

EXAMPLE PR COMMENT:

    ⚠️ Performance Regression Detected
    
    Operation 'data_processing' is 1.5x slower:
    - Baseline: 100ms
    - Current:  150ms
    
    Please investigate or justify this performance change.
    
    [View detailed report](link-to-artifact)

BENEFITS:
  ✅ Catch regressions before merge
  ✅ Enforce performance discipline
  ✅ Document performance changes
  ✅ Maintain performance standards
""")


# ============================================================================
# STEP 4: Performance Budgets
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Performance Budgets")
print("=" * 70)

# Define performance budgets (maximum allowed time)
PERFORMANCE_BUDGETS = {
    "data_processing_pipeline": 2.0,  # Max 2ms
    "mean": 0.5,  # Max 0.5ms
    "std": 1.0,  # Max 1.0ms
    "sum": 0.3,  # Max 0.3ms
}

# Check current performance against budgets
print("\n📊 PERFORMANCE BUDGET ANALYSIS:")

budget_violations = []

for op_name, budget_ms in PERFORMANCE_BUDGETS.items():
    if op_name in new_report["operations"]:
        actual_ms = new_report["operations"][op_name]["avg_time_ms"]
        utilization = (actual_ms / budget_ms) * 100

        status = "✅" if actual_ms <= budget_ms else "❌"
        print(
            f"  {status} {op_name:30s}: {actual_ms:6.2f}ms / {budget_ms:6.2f}ms ({utilization:5.1f}%)"
        )

        if actual_ms > budget_ms:
            budget_violations.append(
                {
                    "operation": op_name,
                    "budget_ms": budget_ms,
                    "actual_ms": actual_ms,
                    "overage": actual_ms - budget_ms,
                }
            )

if budget_violations:
    print("\n❌ BUDGET VIOLATIONS:")
    for violation in budget_violations:
        print(
            f"  {violation['operation']}: Over budget by {violation['overage']:.2f}ms"
        )
else:
    print("\n✅ All operations within performance budgets!")

print("""
💡 PERFORMANCE BUDGETS:

WHAT: Maximum allowed execution time for each operation

WHY:
  - Proactive performance management
  - Clear performance targets
  - Prevents gradual degradation
  - Aligns with user expectations

HOW TO SET BUDGETS:
  1. Start with current performance
  2. Add 20-30% buffer for safety
  3. Tighten over time as you optimize
  4. Base on user requirements (e.g., "page load <2s")

EXAMPLE:
  User requirement: "Process data in <100ms"
  
  Budget breakdown:
    - Data loading:    30ms (30%)
    - Processing:      50ms (50%)
    - Formatting:      20ms (20%)
    Total:            100ms

ENFORCEMENT:
  - CI/CD fails if any operation exceeds budget
  - Requires approval for budget increases
  - Encourages optimization before adding features
""")


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("🎓 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. CREATE BASELINES:
   - Profile current "good" performance
   - Save baseline.json to version control
   - Update when intentional changes occur

2. DETECT REGRESSIONS:
   - Compare new profile to baseline
   - Flag operations >20% slower
   - Severity: HIGH (>2x), MEDIUM (1.2-2x), LOW (<1.2x)

3. CI/CD INTEGRATION:
   - Run performance tests on every PR
   - Automatically detect regressions
   - Fail builds that regress performance
   - Maintain performance discipline

4. PERFORMANCE BUDGETS:
   - Set maximum allowed time per operation
   - Enforce budgets in CI/CD
   - Align with user requirements
   - Proactive vs reactive management

5. WORKFLOW:
   a) Create baseline from main branch
   b) PR triggers performance tests
   c) Compare to baseline
   d) Fail if regression or budget violation
   e) Require justification or fix

TOOLS:
  - cp.export_profile() → Save baseline
  - cp.detect_regressions() → Compare profiles
  - CI/CD integration → Automate enforcement

GOLDEN RULE:
"You can't improve what you don't measure, and you can't
maintain what you don't monitor." Track performance!
""")


# ============================================================================
# WHAT'S NEXT?
# ============================================================================
print("=" * 70)
print("📚 NEXT TUTORIAL")
print("=" * 70)
print("""
You now know how to prevent regressions!
But what about production monitoring?

👉 Run: python 04_production_monitoring.py

You'll learn:
- Safe production profiling strategies
- Sampling and conditional profiling
- Real-time performance alerting
- Production best practices
""")
