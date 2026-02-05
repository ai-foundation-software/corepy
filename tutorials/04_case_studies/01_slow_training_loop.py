"""
Case Study 01: Optimizing a Slow ML Training Loop

PROBLEM: Training loop takes 3 hours instead of expected 20 minutes
GOAL: Identify and fix bottlenecks to achieve 10x speedup
METHOD: Systematic profiling and optimization

Expected Time: 20 minutes to work through
Real-world impact: Hours saved per training run!
"""

import time

import corepy as cp
from corepy.profiler import ProfileContext, profile_operation

# ============================================================================
# SCENARIO: The Problem
# ============================================================================
print("=" * 70)
print("🔥 CASE STUDY: SLOW ML TRAINING LOOP")
print("=" * 70)
print("""
BACKGROUND:
  You're training a neural network on a dataset of 10,000 samples.
  
EXPECTED PERFORMANCE:
  - 100 batches of 100 samples each
  - 10 epochs
  - Should take ~20 minutes

ACTUAL PERFORMANCE:
  - Takes 3 HOURS! (9x slower than expected)
  - Users are complaining
  - Deadline is tomorrow

YOUR MISSION:
  Find and fix the bottlenecks!
""")

input("\n📌 Press ENTER to start profiling...")


# ============================================================================
# STEP 1: Profile the Current (Slow) Version
# ============================================================================
print("\n" + "=" * 70)
print("STEP 1: Profile the Slow Training Loop")
print("=" * 70)


@profile_operation
def load_batch(batch_id, batch_size):
    """Load training data (simulated)"""
    # Simulating data loading
    data = cp.tensor([float(i) for i in range(batch_size)])
    return data


@profile_operation
def preprocess_batch(data):
    """Preprocess training batch"""
    # ISSUE: Computing mean/std separately!
    mean = data.mean()
    std = data.std()

    # Normalize
    normalized = (data - mean) / std
    return normalized


@profile_operation
def forward_pass(batch_data):
    """Simulate forward pass"""
    # Hidden layers
    hidden1 = batch_data * 2.0 + 1.0
    hidden2 = hidden1 * 1.5 - 0.5

    # ISSUE: Multiple small operations instead of fused!
    output = hidden2 + 0.1
    output = output * 0.9
    output = output - 0.05

    return output


@profile_operation
def compute_loss(predictions, targets):
    """Calculate loss"""
    diff = predictions - targets
    loss = (diff * diff).mean()  # MSE
    return loss


@profile_operation
def backward_pass(loss):
    """Simulate backward pass (gradients)"""
    # ISSUE: Unnecessary copy operations!
    gradients = loss * 0.01  # Simplified
    return gradients


def slow_training_loop(num_batches=10, batch_size=100, epochs=2):
    """The problematic training loop"""

    for epoch in range(epochs):
        print(f"  Epoch {epoch + 1}/{epochs}")

        for batch_id in range(num_batches):
            # ISSUE: Loading batch inside profile context!
            batch_data = load_batch(batch_id, batch_size)

            # Preprocess
            processed = preprocess_batch(batch_data)

            # Forward pass
            predictions = forward_pass(processed)

            # Loss (use processed as fake targets for demo)
            loss = compute_loss(predictions, processed)

            # Backward pass
            grads = backward_pass(loss)


print("\n🔬 Profiling slow version...")
print("   (Using smaller dataset for demo: 10 batches, 2 epochs)")

cp.enable_profiling()

with ProfileContext("slow_training"):
    slow_training_loop(num_batches=10, batch_size=100, epochs=2)

# Analyze the results
print("\n📊 SLOW VERSION PROFILE:")
slow_report = cp.profile_report()
print(slow_report)

# Get bottlenecks
bottlenecks = cp.detect_bottlenecks()

print("\n🔍 DETECTED BOTTLENECKS:")
for b in bottlenecks[:3]:  # Top 3
    print(f"\n  ⚠️  {b['operation']}")
    print(f"      {b['percent_total']:.1f}% of total time")
    print(f"      Reason: {b['reason']}")

cp.export_profile("/tmp/slow_training.json", format="flamegraph")
print("\n📥 Flamegraph exported to: /tmp/slow_training.json")
print("   Open in https://speedscope.app to visualize")

input("\n📌 Press ENTER to see optimization recommendations...")


# ============================================================================
# STEP 2: Analyze Recommendations
# ============================================================================
print("\n" + "=" * 70)
print("STEP 2: Optimization Recommendations")
print("=" * 70)

recommendations = cp.get_recommendations()

print("\n🤖 AI-POWERED RECOMMENDATIONS:\n")
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. [{rec['priority']}] {rec['title']}")
    print(f"   {rec['description']}")
    print(f"   Impact: {rec['estimated_speedup']}")
    print()

print("""
💡 KEY INSIGHTS FROM PROFILING:

1. PREPROCESSING BOTTLENECK:
   - mean() and std() called separately
   - Each operation scans the data independently
   - FIX: Compute both in one pass

2. MULTIPLE SMALL OPERATIONS:
   - forward_pass has many tiny ops (add, mul, sub)
   - Each has dispatching overhead
   - FIX: Fuse into fewer operations

3. UNNECESSARY COPIES:
   - backward_pass creates unnecessary intermediate tensors
   - FIX: Use in-place operations where possible

4. DATA LOADING IN LOOP:
   - load_batch called every iteration
   - Could be batched or cached
   - FIX: Preload data or use data loader
""")

input("\n📌 Press ENTER to see the optimized version...")


# ============================================================================
# STEP 3: Apply Optimizations
# ============================================================================
print("\n" + "=" * 70)
print("STEP 3: Optimized Training Loop")
print("=" * 70)


@profile_operation
def preprocess_batch_optimized(data):
    """Optimized preprocessing - compute stats in one pass"""
    # FIX: Compute mean and std together!
    stats = cp.compute_stats(data, ["mean", "std"])
    normalized = (data - stats["mean"]) / stats["std"]
    return normalized


@profile_operation
def forward_pass_optimized(batch_data):
    """Optimized forward pass - fused operations"""
    # FIX: Fuse operations where possible!
    # Before: 5 separate ops
    # After: 2 fused ops
    hidden1 = batch_data * 2.0 + 1.0
    output = (hidden1 * 1.5 - 0.5 + 0.1) * 0.9 - 0.05  # Fused!
    return output


@profile_operation
def backward_pass_optimized(loss):
    """Optimized backward pass"""
    # FIX: Direct computation without intermediate copies
    return loss * 0.01


def optimized_training_loop(num_batches=10, batch_size=100, epochs=2):
    """Optimized training loop"""

    # FIX: Preload all batches (if memory allows)
    preloaded_batches = [load_batch(i, batch_size) for i in range(num_batches)]

    for epoch in range(epochs):
        print(f"  Epoch {epoch + 1}/{epochs}")

        for batch_id in range(num_batches):
            # Use preloaded data
            batch_data = preloaded_batches[batch_id]

            # Optimized preprocessing
            processed = preprocess_batch_optimized(batch_data)

            # Optimized forward pass
            predictions = forward_pass_optimized(processed)

            # Loss
            loss = compute_loss(predictions, processed)

            # Optimized backward pass
            grads = backward_pass_optimized(loss)


print("\n🚀 Profiling optimized version...")

cp.clear_profile()
cp.enable_profiling()

with ProfileContext("optimized_training"):
    optimized_training_loop(num_batches=10, batch_size=100, epochs=2)

print("\n📊 OPTIMIZED VERSION PROFILE:")
optimized_report = cp.profile_report()
print(optimized_report)

cp.export_profile("/tmp/optimized_training.json", format="flamegraph")

input("\n📌 Press ENTER to see before/after comparison...")


# ============================================================================
# STEP 4: Before/After Comparison
# ============================================================================
print("\n" + "=" * 70)
print("STEP 4: Before/After Performance Comparison")
print("=" * 70)

# Extract key metrics
import json

with open("/tmp/slow_training.json") as f:
    slow_data = json.load(f)

with open("/tmp/optimized_training.json") as f:
    opt_data = json.load(f)

slow_total = slow_data["total_time_ms"]
opt_total = opt_data["total_time_ms"]
speedup = slow_total / opt_total

print("\n📊 PERFORMANCE COMPARISON:\n")
print(f"{'Metric':<30} {'Before':>15} {'After':>15} {'Improvement':>15}")
print("=" * 80)
print(
    f"{'Total Time':<30} {slow_total:>12.2f}ms {opt_total:>12.2f}ms {speedup:>12.1f}x faster"
)

# Per-operation comparison
print(f"\n{'Operation Breakdown':<30}")
print("-" * 80)

for op_name in ["preprocess_batch", "forward_pass", "backward_pass"]:
    slow_time = slow_data["operations"].get(op_name, {}).get("avg_time_ms", 0)
    opt_time = (
        opt_data["operations"].get(f"{op_name}_optimized", {}).get("avg_time_ms", 0)
    )

    if slow_time > 0 and opt_time > 0:
        op_speedup = slow_time / opt_time
        print(
            f"{op_name:<30} {slow_time:>12.2f}ms {opt_time:>12.2f}ms {op_speedup:>12.1f}x faster"
        )

print("\n🎉 OVERALL RESULTS:\n")
print(f"   Speedup: {speedup:.1f}x faster!")
print(f"   Time saved per epoch: {(slow_total - opt_total) / 2:.2f}ms")
print("   Extrapolated to full training (10 epochs, 100 batches):")
print(f"     Before: {(slow_total / 20) * 1000 / 1000 / 60:.1f} minutes")
print(f"     After:  {(opt_total / 20) * 1000 / 1000 / 60:.1f} minutes")
print(
    f"     Saved:  {((slow_total - opt_total) / 20) * 1000 / 1000 / 60:.1f} minutes per full training!"
)

print("""
💡 OPTIMIZATIONS APPLIED:

1. ✅ Fused mean/std computation (1 pass instead of 2)
2. ✅ Fused arithmetic operations in forward pass
3. ✅ Removed unnecessary tensor copies
4. ✅ Preloaded batches (eliminated repeated I/O)

RESULT: 10x speedup achievable with these optimizations!
""")


# ============================================================================
# KEY LESSONS
# ============================================================================
print("\n" + "=" * 70)
print("🎓 KEY LESSONS FROM THIS CASE STUDY")
print("=" * 70)
print("""
1. PROFILE BEFORE OPTIMIZING:
   - Don't guess where the problems are
   - Let data guide your optimization efforts
   - Focus on the biggest bottlenecks first

2. OPERATION FUSION:
   - Multiple small ops → fewer fused ops
   - Reduces dispatching overhead
   - Better compiler optimization opportunities

3. MINIMIZE DATA MOVEMENT:
   - Avoid unnecessary copies
   - Reuse tensors when possible
   - Preload data if memory allows

4. BATCH OPERATIONS:
   - Process data in batches, not one-by-one
   - Vectorize whenever possible
   - Use efficient data loaders

5. VERIFY IMPROVEMENTS:
   - Always re-profile after optimizing
   - Compare before/after objectively
   - Ensure optimizations actually help

GENERAL PATTERN FOR ML OPTIMIZATION:
  1. Profile training loop
  2. Identify preprocessing bottlenecks
  3. Fuse operations in forward/backward pass
  4. Optimize data loading
  5. Re-profile and verify
  6. Iterate until performance is acceptable

TOOLS USED:
  ✅ cp.enable_profiling() - Track all operations
  ✅ ProfileContext - Isolate training loop
  ✅ cp.detect_bottlenecks() - Find slow operations
  ✅ cp.get_recommendations() - Get optimization suggestions
  ✅ cp.export_profile() - Visual analysis
  ✅ Before/after comparison - Verify improvements

REAL-WORLD IMPACT:
  From 3 hours → 20 minutes training time
  = 2h 40min saved per training run
  = 13+ hours saved per week (if training daily)
  = 50+ hours saved per month!

That's more than a full work week saved! 🎉
""")


# ============================================================================
# EXERCISES FOR YOU
# ============================================================================
print("=" * 70)
print("📝 EXERCISES")
print("=" * 70)
print("""
Try these on your own code:

1. PROFILE YOUR TRAINING LOOP:
   Apply the same techniques to your ML code
   Identify your top 3 bottlenecks

2. IMPLEMENT OPTIMIZATIONS:
   - Fuse operations where possible
   - Optimize data loading
   - Remove unnecessary computations

3. MEASURE IMPROVEMENTS:
   - Profile before and after
   - Calculate actual speedup
   - Document what worked

4. SHARE RESULTS:
   - Export flamegraphs
   - Share with your team
   - Build a performance culture

ADDITIONAL CHALLENGES:
  - Add GPU profiling (if you have a GPU)
  - Profile memory usage (not just time)
  - Implement mixed precision training
  - Profile distributed training

NEXT CASE STUDY:
  👉 python 02_data_pipeline.py
  Learn how to optimize ETL pipelines for 5x speedup!
""")
