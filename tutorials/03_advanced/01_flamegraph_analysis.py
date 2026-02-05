"""
Tutorial 03-1: Flamegraph Analysis

WHAT: Generate visual performance flamegraphs
WHY: See the entire call hierarchy and bottlenecks at a glance
HOW: Export to Speedscope format and analyze visually

Expected Time: 10 minutes
"""

import corepy as cp
from corepy.profiler import ProfileContext, profile_operation

# ============================================================================
# INTRODUCTION: What is a Flamegraph?
# ============================================================================
print("=" * 70)
print("🔥 WHAT IS A FLAMEGRAPH?")
print("=" * 70)
print("""
A flamegraph is a visualization where:
  - X-axis (width): Time spent (wider = slower)
  - Y-axis (height): Call stack depth (taller = deeper nesting)
  - Color: Different functions (for visual distinction)

EXAMPLE:
  ┌─────────────────────────────────────┐
  │        full_pipeline (100%)         │  ← Top level
  └─────────────┬───────────────────────┘
                │
  ┌─────────────┴──────┬────────────────┐
  │  preprocess (30%)  │  compute (70%) │  ← Second level
  └─────────┬──────────┴────────────────┘
            │
  ┌─────────┴──────┬────────┐
  │  mean (10%)    │ std(20%)│              ← Third level
  └────────────────┴─────────┘

WIDTH tells you WHERE time is spent (compute = 70% of total!)
HEIGHT tells you HOW functions call each other

One glance = understand entire performance profile!
""")


# ============================================================================
# EXAMPLE 1: Generate Your First Flamegraph
# ============================================================================
print("=" * 70)
print("EXAMPLE 1: Generating a Flamegraph")
print("=" * 70)


# Create a realistic nested call structure
@profile_operation
def load_data():
    """Data loading"""
    return cp.tensor([float(i) for i in range(1000)])


@profile_operation
def normalize(data):
    """Normalization step"""
    mean = data.mean()
    std = data.std()
    return (data - mean) / std


@profile_operation
def compute_features(data):
    """Feature engineering"""
    # Multiple operations
    sum_val = data.sum()
    prod = data * data
    stats = prod.mean()
    return stats


@profile_operation
def process_pipeline(data):
    """Main processing pipeline"""
    normalized = normalize(data)
    features = compute_features(normalized)
    return features


@profile_operation
def full_workflow():
    """Complete workflow"""
    data = load_data()
    result = process_pipeline(data)
    return result


print("\n1. Running profiled code...")
cp.enable_profiling()

result = full_workflow()

print("✅ Code execution complete")

# Export as flamegraph
output_path = "/tmp/corepy_flamegraph.json"
cp.export_profile(output_path, format="flamegraph")

print(f"\n2. Flamegraph exported to: {output_path}")
print("\n3. TO VIEW:")
print("   a. Open https://speedscope.app in your browser")
print("   b. Click 'Browse' and select the JSON file")
print("   c. Explore the interactive flamegraph!")

print("""
💡 WHAT YOU'LL SEE:

In Speedscope, you'll see:
  - full_workflow at the top (entire width = 100% of time)
  - load_data and process_pipeline as children
  - normalize and compute_features under process_pipeline
  - Individual operations (mean, std, sum, mul) at the bottom

CLICK on any box to see:
  - Function name
  - Time spent (absolute and percentage)
  - Number of calls
  - Ancestors (who called this)
  - Descendants (what this called)
""")


# ============================================================================
# EXAMPLE 2: Reading a Flamegraph - Finding Bottlenecks
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: How to Read a Flamegraph")
print("=" * 70)

cp.clear_profile()
cp.enable_profiling()


# Simulate a bottleneck scenario
@profile_operation
def fast_operation_1(data):
    return data + 1.0


@profile_operation
def fast_operation_2(data):
    return data * 2.0


@profile_operation
def slow_bottleneck(data):
    """This is intentionally slow"""
    result = data.copy()
    for _ in range(100):
        result = result * 1.001
    return result


@profile_operation
def mixed_pipeline(data):
    step1 = fast_operation_1(data)
    step2 = fast_operation_2(step1)
    step3 = slow_bottleneck(step2)  # ← This will be WIDE in the flamegraph!
    return step3


data = cp.tensor([float(i) for i in range(500)])
result = mixed_pipeline(data)

cp.export_profile("/tmp/bottleneck_flamegraph.json", format="flamegraph")

print("""
🔍 FINDING BOTTLENECKS IN THE FLAMEGRAPH:

1. LOOK FOR WIDTH:
   - The WIDEST boxes are the slowest operations
   - In this example, 'slow_bottleneck' will be very wide
   - fast_operation_1 and fast_operation_2 will be narrow

2. CHECK THE STACK:
   - Tall stacks = deep function nesting
   - Wide boxes at bottom = low-level bottlenecks
   - Wide boxes at top = high-level bottlenecks

3. ZOOM IN:
   - Click any box to focus on it
   - Speedscope shows you:
     * Exact time in milliseconds
     * Percentage of parent's time
     * Number of calls

EXAMPLE INSIGHTS:
  "slow_bottleneck takes 95% of mixed_pipeline's time"
  → THIS is where optimization effort should go!
  
  "fast_operation_1 only takes 2% of time"
  → Don't waste time optimizing this
""")


# ============================================================================
# EXAMPLE 3: Comparing Before/After with Flamegraphs
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Before/After Optimization")
print("=" * 70)

# BEFORE: Unoptimized version
cp.clear_profile()
cp.enable_profiling()

with ProfileContext("before_optimization"):
    data = cp.tensor([float(i) for i in range(1000)])

    # Inefficient: multiple passes over data
    mean_val = data.mean()
    std_val = data.std()
    sum_val = data.sum()

    result = (data - mean_val) / std_val + sum_val

cp.export_profile("/tmp/before_optimization.json", format="flamegraph")


# AFTER: Optimized version
cp.clear_profile()
cp.enable_profiling()

with ProfileContext("after_optimization"):
    data = cp.tensor([float(i) for i in range(1000)])

    # Efficient: compute stats in one pass
    stats = cp.compute_stats(data, ["mean", "std", "sum"])
    result = (data - stats["mean"]) / stats["std"] + stats["sum"]

cp.export_profile("/tmp/after_optimization.json", format="flamegraph")

print("""
📊 COMPARING FLAMEGRAPHS:

1. Open both files in Speedscope (use tabs)
2. Switch between them to compare

BEFORE:
  - You'll see separate boxes for mean(), std(), sum()
  - Each is a separate pass over the data
  - Total width shows cumulative time

AFTER:
  - compute_stats() is a single box
  - Less total width = faster!
  - Visual proof of optimization

TYPICAL IMPROVEMENTS:
  - Fewer function calls (shorter stacks)
  - Narrower boxes (less time)
  - Simpler call graphs (less complexity)
""")


# ============================================================================
# EXAMPLE 4: Advanced Flamegraph Features
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Advanced Speedscope Features")
print("=" * 70)

print("""
SPEEDSCOPE FEATURES:

1. VIEW MODES:
   - Time Order: See operations chronologically
   - Left Heavy: Group by call stack (easier to read)
   - Sandwich: Show callers and callees

2. SEARCH:
   - Press '/' to search for function names
   - Highlights matching boxes
   - Example: Search "matmul" to find all matrix multiplications

3. STATISTICS:
   - Click a box → see detailed timing stats
   - Self time: Time spent in function itself
   - Total time: Time including all

 callees

4. EXPORT:
   - Share flamegraphs with teammates
   - Save snapshots for comparison
   - Attach to performance bug reports

5. KEYBOARD SHORTCUTS:
   - '+' / '-': Zoom in/out
   - 'w' / 's': Zoom to fit
   - '/' : Search
   - 'Esc': Clear selection
""")


# ============================================================================
# PRACTICAL WORKFLOW
# ============================================================================
print("\n" + "=" * 70)
print("🔧 PRACTICAL WORKFLOW")
print("=" * 70)
print("""
STEP-BY-STEP OPTIMIZATION WORKFLOW:

1. PROFILE & EXPORT:
   cp.enable_profiling()
   run_your_code()
   cp.export_profile("profile.json", format="flamegraph")

2. ANALYZE IN SPEEDSCOPE:
   - Open https://speedscope.app
   - Load profile.json
   - Find the widest boxes (bottlenecks)

3. OPTIMIZE BOTTLENECKS:
   - Focus on functions taking >20% of time
   - Ignore functions taking <5% of time

4. RE-PROFILE:
   - Generate new flamegraph after changes
   - Open both (before/after) in separate tabs
   - Verify improvement visually

5. ITERATE:
   - Repeat until performance is acceptable
   - Keep flamegraphs for documentation

WHEN TO USE FLAMEGRAPHS:
  ✅ Complex call hierarchies (many nested functions)
  ✅ Unknown bottlenecks (exploratory profiling)
  ✅ Explaining performance to others (visual is clear)
  ✅ Code reviews (attach to performance PRs)
""")


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("🎓 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. FLAMEGRAPHS SHOW:
   - Width = time spent (wider = slower)
   - Height = call stack depth
   - One glance = entire performance profile

2. EXPORT FORMAT:
   cp.export_profile(path, format="flamegraph")
   # Compatible with Speedscope.app

3. READING FLAMEGRAPHS:
   - Look for WIDE boxes (bottlenecks)
   - Check stack depth (nesting complexity)
   - Compare before/after visually

4. TOOLS:
   - Speedscope: https://speedscope.app (best!)
   - Chrome Tracing: chrome://tracing (alternative)

5. WORKFLOW:
   Profile → Export → Visualize → Optimize → Repeat

GOLDEN RULE:
"One flamegraph is worth a thousand text reports"
Visual > Text for understanding performance!
""")


# ============================================================================
# WHAT'S NEXT?
# ============================================================================
print("=" * 70)
print("📚 NEXT TUTORIAL")
print("=" * 70)
print("""
You can now visualize performance with flamegraphs!
But what about integrating with other tools?

👉 Run: python 02_export_integration.py

You'll learn:
- Export to multiple formats (JSON, CSV, Chrome Tracing)
- Integrate with existing tooling
- Automate performance analysis
- Share profiling data with teams
""")
