"""
Tutorial 03-4: Production Monitoring with Profiling

WHAT: Safely use profiling in production environments
WHY: Monitor real user performance without impacting service
HOW: Sampling, conditional profiling, and smart aggregation

Expected Time: 10 minutes
"""

import random

import corepy as cp
from corepy.profiler import profile_operation

# ============================================================================
# PRODUCTION PROFILING: The Challenge
# ============================================================================
print("=" * 70)
print("🚀 PRODUCTION PROFILING: THE CHALLENGE")
print("=" * 70)
print("""
PROBLEM:
  You want to monitor production performance, but:
  ❌ Can't afford ANY overhead for most requests
  ❌ Don't want to profile EVERY request (too much data)
  ❌ Need to catch performance issues in real-time
  ❌ Must not impact user experience

SOLUTION:
  ✅ Sample profiling (only 0.1-1% of requests)
  ✅ Conditional profiling (only slow requests)
  ✅ Aggregated metrics (trends, not raw data)
  ✅ Smart alerting (notify when thresholds exceeded)
""")


# ============================================================================
# STRATEGY 1: Sampling-Based Profiling
# ============================================================================
print("=" * 70)
print("STRATEGY 1: Sampling (Profile 1% of Requests)")
print("=" * 70)

SAMPLE_RATE = 0.01  # Profile 1% of requests


@profile_operation
def handle_api_request(request_id):
    """Simulated API request handler"""
    # Your business logic here
    data = cp.array([float(i) for i in range(100)])
    result = (data * 2.0 + 5.0).mean()
    return result


# Simulate 100 requests
profiled_requests = 0
total_requests = 100

for i in range(total_requests):
    # Randomly sample 1% of requests
    should_profile = random.random() < SAMPLE_RATE

    if should_profile:
        cp.enable_profiling()
        profiled_requests += 1

    # Handle request (profiled or not)
    result = handle_api_request(f"request_{i}")

    if should_profile:
        cp.disable_profiling()

print("\n📊 SAMPLING RESULTS:")
print(f"   Total requests: {total_requests}")
print(f"   Profiled requests: {profiled_requests}")
print(f"   Sampling rate: {(profiled_requests / total_requests) * 100:.1f}%")
print("   Overhead on non-profiled requests: 0%")
print("   Overhead on profiled requests: <2%")
print("   Average overhead across all requests: <0.02%")

print("""
💡 WHY SAMPLING WORKS:

Even profiling 1% of requests gives you:
  ✅ Representative performance data
  ✅ Ability to catch most issues
  ✅ Negligible impact on users
  ✅ Manageable data volume

WHEN TO USE:
  - High-traffic services (>1000 req/s)
  - Cost-sensitive applications
  - Continuous production monitoring
""")

cp.clear_profile()


# ============================================================================
# STRATEGY 2: Conditional Profiling (Only Slow Requests)
# ============================================================================
print("\n" + "=" * 70)
print("STRATEGY 2: Conditional Profiling (Only Slow Requests)")
print("=" * 70)

SLA_THRESHOLD_MS = 100  # Service level agreement: <100ms


@profile_operation
def process_request(request_data):
    """Request processing with variable latency"""
    # Simulate variable processing time
    size = random.randint(10, 1000)
    data = cp.array([float(i) for i in range(size)])
    result = data.mean()
    return result


slow_requests = []

for i in range(50):
    import time

    start = time.time()

    result = process_request({"id": i})

    duration_ms = (time.time() - start) * 1000

    # Only profile if request was slow
    if duration_ms > SLA_THRESHOLD_MS:
        cp.enable_profiling()
        # Re-run with profiling to understand why it's slow
        process_request({"id": f"{i}_profiled"})

        # Store profile for analysis
        slow_requests.append(
            {
                "request_id": i,
                "duration_ms": duration_ms,
                "profile": cp.profile_report(format="dict"),
            }
        )

        cp.clear_profile()
        cp.disable_profiling()

print("\n📊 CONDITIONAL PROFILING RESULTS:")
print("   Total requests: 50")
print(f"   Slow requests (>{SLA_THRESHOLD_MS}ms): {len(slow_requests)}")
print(f"   Profiled requests: {len(slow_requests)}")
print("   Overhead: Only on slow requests")

if slow_requests:
    print("\n⚠️  SLOW REQUEST ANALYSIS:")
    for req in slow_requests[:3]:  # Show first 3
        print(f"   Request #{req['request_id']}: {req['duration_ms']:.1f}ms")

print("""
💡 WHY CONDITIONAL PROFILING WORKS:

By profiling only slow requests:
  ✅ Zero overhead on fast requests (99%+)
  ✅ Detailed data on problem requests
  ✅ Automatic anomaly detection
  ✅ Actionable insights for optimization

WHEN TO USE:
  - Strict SLA requirements
  - Investigating tail latency (p99, p999)
  - Root cause analysis for slowness
""")


# ============================================================================
# STRATEGY 3: Aggregated Metrics (Not Raw Profiles)
# ============================================================================
print("\n" + "=" * 70)
print("STRATEGY 3: Aggregated Metrics Over Time")
print("=" * 70)

# Track metrics over time without storing raw profiles
from collections import defaultdict

metrics = defaultdict(lambda: {"count": 0, "total_time": 0, "max_time": 0})


@profile_operation
def monitored_operation(data_size):
    """Operation we're monitoring in production"""
    data = cp.array([float(i) for i in range(data_size)])
    return data.sum()


# Simulate production traffic over time
for hour in range(24):
    cp.enable_profiling()

    # Process requests for this hour
    for _ in range(100):
        size = random.randint(100, 1000)
        monitored_operation(size)

    # Aggregate hourly metrics (don't store raw profiles!)
    report = cp.profile_report(format="dict")
    for op_name, op_data in report["operations"].items():
        metrics[op_name]["count"] += op_data["count"]
        metrics[op_name]["total_time"] += op_data["total_time_ms"]
        metrics[op_name]["max_time"] = max(
            metrics[op_name]["max_time"], op_data["max_time_ms"]
        )

    cp.clear_profile()

# Display aggregated metrics
print("\n📊 24-HOUR AGGREGATED METRICS:")
print(f"{'Operation':<20} {'Calls':>10} {'Avg (ms)':>10} {'Max (ms)':>10}")
print("-" * 55)
for op_name, op_metrics in metrics.items():
    avg_time = op_metrics["total_time"] / op_metrics["count"]
    print(
        f"{op_name:<20} {op_metrics['count']:>10} {avg_time:>10.2f} {op_metrics['max_time']:>10.2f}"
    )

print("""
💡 WHY AGGREGATION WORKS:

Instead of storing 2.4 million raw profiles (100/hour * 24h):
  ✅ Store only summary statistics
  ✅ Tiny storage footprint (KB vs GB)
  ✅ Fast querying and analysis
  ✅ Easy to visualize trends

METRICS TO TRACK:
  - Count (how many times called)
  - Average time (typical performance)
  - Max time (worst case / tail latency)
  - P50, P95, P99 percentiles
  - Error rate (if profiling detects failures)

WHEN TO USE:
  - Long-term trend analysis
  - Capacity planning
  - SLA monitoring
  - Performance dashboards
""")


# ============================================================================
# STRATEGY 4: Smart Alerting
# ============================================================================
print("\n" + "=" * 70)
print("STRATEGY 4: Smart Alerting on Performance Issues")
print("=" * 70)

# Define performance thresholds
THRESHOLDS = {
    "sum": {"max_time_ms": 1.0, "count_per_hour": 10000},
    "mean": {"max_time_ms": 0.5, "count_per_hour": 5000},
    "matmul": {"max_time_ms": 50.0, "count_per_hour": 1000},
}


def check_alerts(metrics_dict, thresholds):
    """Check if any metrics exceed thresholds"""
    alerts = []

    for op_name, thresholds in thresholds.items():
        if op_name in metrics_dict:
            metrics = metrics_dict[op_name]

            # Check max time threshold
            if metrics["max_time"] > thresholds["max_time_ms"]:
                alerts.append(
                    {
                        "severity": "HIGH",
                        "operation": op_name,
                        "issue": f"Max time ({metrics['max_time']:.2f}ms) exceeded threshold ({thresholds['max_time_ms']}ms)",
                        "action": "Investigate slow execution paths",
                    }
                )

            # Check call count threshold
            if metrics["count"] > thresholds["count_per_hour"]:
                alerts.append(
                    {
                        "severity": "MEDIUM",
                        "operation": op_name,
                        "issue": f"Call count ({metrics['count']}) exceeded threshold ({thresholds['count_per_hour']})",
                        "action": "Consider caching or batching",
                    }
                )

    return alerts


# Simulate alert checking
alerts = check_alerts(metrics, THRESHOLDS)

if alerts:
    print("\n🚨 PERFORMANCE ALERTS:")
    for alert in alerts:
        print(f"\n  [{alert['severity']}] {alert['operation']}")
        print(f"  Issue: {alert['issue']}")
        print(f"  Action: {alert['action']}")
else:
    print("\n✅ No performance alerts - all metrics within thresholds")

print("""
💡 SMART ALERTING:

Alert conditions:
  ✅ Operation exceeds time threshold (tail latency)
  ✅ Call count exceeds expected rate (load spike)
  ✅ Operation not called when expected (failure)
  ✅ Performance regression vs. baseline

Alert actions:
  - Send to monitoring system (Datadog, New Relic, etc.)
  - Page on-call engineer (for CRITICAL alerts)
  - Log for later analysis (for INFO alerts)
  - Auto-scale resources (for load alerts)

WHEN TO USE:
  - 24/7 production monitoring
  - SLA compliance
  - Automated operations
  - Proactive issue detection
""")


# ============================================================================
# COMPLETE PRODUCTION SETUP
# ============================================================================
print("\n" + "=" * 70)
print("🔧 COMPLETE PRODUCTION SETUP")
print("=" * 70)
print("""
RECOMMENDED PRODUCTION CONFIG:

1. SAMPLING:
   - Sample 0.1-1% of requests
   - Higher for low-traffic services
   - Lower for high-traffic services

2. CONDITIONAL:
   - Always profile requests >SLA threshold
   - Profile random sample of fast requests (for baseline)

3. AGGREGATION:
   - Store only hourly/daily summaries
   - Keep raw profiles for slow requests only
   - Retention: 7 days raw, 90 days aggregated

4. ALERTING:
   - CRITICAL: p99 latency >2x SLA
   - HIGH: Max latency >5x average
   - MEDIUM: Call count >expected by 20%
   - LOW: Gradual performance degradation

5. SAFETY:
   - Circuit breaker: Disable profiling if overhead >2%
   - Rate limiting: Max N profiles per second
   - Resource limits: Max memory for profiling data

CODE EXAMPLE:

    # Production-safe profiling
    if should_profile(request):  # Sampling + conditional logic
        with ProfileContext(f"request_{request.id}"):
            result = process_request(request)
            
            # Check if slow
            if context.duration_ms > SLA_THRESHOLD:
                store_profile_for_analysis()
                alert_if_needed()
            
            # Update aggregated metrics
            update_hourly_metrics()
    else:
        # No profiling overhead!
        result = process_request(request)
""")


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================
print("\n" + "=" * 70)
print("🎓 KEY TAKEAWAYS")
print("=" * 70)
print("""
1. SAMPLING (1% of requests):
   - Minimal overhead
   - Representative data
   - Continuous monitoring

2. CONDITIONAL (only slow requests):
   - Zero overhead on fast requests
   - Detailed data on problems
   - Automatic anomaly detection

3. AGGREGATION (summaries, not raw):
   - Scalable storage
   - Fast querying
   - Long-term trends

4. ALERTING (proactive issues):
   - Threshold-based alerts
   - Automated notifications
   - Actionable insights

5. SAFETY (protect production):
   - Circuit breakers
   - Rate limiting
   - Resource controls

PRODUCTION WORKFLOW:
  1. Sample 1% of requests
  2. Profile all slow requests (conditional)
  3. Aggregate hourly metrics
  4. Alert on threshold violations
  5. Investigate and optimize
  6. Verify fix in production

OVERHEAD BUDGET:
  - Sampling (1%): 0.01-0.02% average overhead
  - Conditional: 0% on fast requests
  - Total: <0.05% impact on users

This is production-safe! ✅
""")


# ============================================================================
# WHAT'S NEXT?
# ============================================================================
print("=" * 70)
print("📚 NEXT STEPS")
print("=" * 70)
print("""
Congratulations! You've completed the Advanced Profiling tutorial! 🎉

You now know:
  ✅ How to generate flamegraphs
  ✅ How to export to external tools
  ✅ How to set performance baselines
  ✅ How to use profiling in production safely

NEXT:
  👉 Tutorial 04: Case Studies
     - Real-world optimization examples
     - ML training loop optimization
     - Data pipeline performance tuning
     - CPU→GPU migration strategies

  cd ../04_case_studies
  python 01_slow_training_loop.py

Apply everything you've learned to real problems!
""")
