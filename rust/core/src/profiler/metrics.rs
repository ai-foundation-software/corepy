//! Performance profiling metrics and data structures
//!
//! This module defines the core data types for the profiling system:
//! - OperationEvent: Individual operation timing records
//! - OperationMetrics: Aggregated statistics for an operation
//! - ProfileReport: Complete profiling session report

use serde::{Deserialize, Serialize};

/// Represents a single profiled operation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationEvent {
    /// Name of the operation (e.g., "add", "matmul", "sum")
    pub operation: String,

    /// Backend that executed the operation ("CPU", "GPU", etc.)
    pub backend: String,

    /// Number of elements in the operation
    pub data_size: usize,

    /// Start timestamp (microseconds since epoch)
    pub start_time_us: u64,

    /// End timestamp (microseconds since epoch)
    pub end_time_us: u64,

    /// Optional context/section name (for ProfileContext)
    pub context: Option<String>,
}

impl OperationEvent {
    /// Calculate the duration of this operation in microseconds
    pub fn duration_us(&self) -> u64 {
        self.end_time_us.saturating_sub(self.start_time_us)
    }

    /// Calculate the duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        self.duration_us() as f64 / 1000.0
    }
}

/// Aggregated metrics for a specific operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    /// Operation name
    pub operation: String,

    /// Number of times this operation was called
    pub count: usize,

    /// Total time spent in this operation (milliseconds)
    pub total_time_ms: f64,

    /// Average time per call (milliseconds)
    pub avg_time_ms: f64,

    /// Minimum time observed (milliseconds)
    pub min_time_ms: f64,

    /// Maximum time observed (milliseconds)
    pub max_time_ms: f64,

    /// Most common backend used
    pub primary_backend: String,

    /// Percentage of total execution time
    pub percent_total: f64,
}

impl OperationMetrics {
    /// Create metrics from a list of events for a single operation
    pub fn from_events(operation: &str, events: &[OperationEvent], total_time_ms: f64) -> Self {
        let count = events.len();

        if count == 0 {
            return Self {
                operation: operation.to_string(),
                count: 0,
                total_time_ms: 0.0,
                avg_time_ms: 0.0,
                min_time_ms: 0.0,
                max_time_ms: 0.0,
                primary_backend: "unknown".to_string(),
                percent_total: 0.0,
            };
        }

        let durations_ms: Vec<f64> = events.iter().map(|e| e.duration_ms()).collect();

        let total = durations_ms.iter().sum::<f64>();
        let avg = total / count as f64;
        let min = durations_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = durations_ms
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Find most common backend
        let mut backend_counts = std::collections::HashMap::new();
        for event in events {
            *backend_counts.entry(&event.backend).or_insert(0) += 1;
        }
        let primary_backend = backend_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(backend, _)| backend.clone())
            .unwrap_or_else(|| "unknown".to_string());

        let percent = if total_time_ms > 0.0 {
            (total / total_time_ms) * 100.0
        } else {
            0.0
        };

        Self {
            operation: operation.to_string(),
            count,
            total_time_ms: total,
            avg_time_ms: avg,
            min_time_ms: min,
            max_time_ms: max,
            primary_backend,
            percent_total: percent,
        }
    }
}

/// Complete profiling report for a session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReport {
    /// Session metadata
    pub metadata: SessionMetadata,

    /// Metrics for each operation
    pub operations: std::collections::HashMap<String, OperationMetrics>,

    /// Total execution time across all operations (milliseconds)
    pub total_time_ms: f64,

    /// Number of operations profiled
    pub operation_count: usize,
}

/// Metadata about the profiling session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Unique session identifier
    pub session_id: String,

    /// Timestamp when profiling started
    pub start_timestamp: String,

    /// Corepy version
    pub version: String,

    /// Optional context filter (if report is for specific context)
    pub context: Option<String>,
}

impl ProfileReport {
    /// Create a new empty report
    pub fn new(session_id: String, context: Option<String>) -> Self {
        let now = chrono::Utc::now();

        Self {
            metadata: SessionMetadata {
                session_id,
                start_timestamp: now.to_rfc3339(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                context,
            },
            operations: std::collections::HashMap::new(),
            total_time_ms: 0.0,
            operation_count: 0,
        }
    }

    /// Build a report from a list of events
    pub fn from_events(events: &[OperationEvent], context_filter: Option<&str>) -> Self {
        // Filter events by context if specified
        let filtered_events: Vec<&OperationEvent> = if let Some(ctx) = context_filter {
            events
                .iter()
                .filter(|e| e.context.as_deref() == Some(ctx))
                .collect()
        } else {
            events.iter().collect()
        };

        if filtered_events.is_empty() {
            return Self::new(
                uuid::Uuid::new_v4().to_string(),
                context_filter.map(String::from),
            );
        }

        // Calculate total time
        let total_time_ms: f64 = filtered_events.iter().map(|e| e.duration_ms()).sum();

        // Group events by operation
        let mut operation_groups: std::collections::HashMap<String, Vec<OperationEvent>> =
            std::collections::HashMap::new();

        for event in filtered_events {
            operation_groups
                .entry(event.operation.clone())
                .or_default()
                .push((*event).clone());
        }

        // Create metrics for each operation
        let operations: std::collections::HashMap<String, OperationMetrics> = operation_groups
            .iter()
            .map(|(op_name, events)| {
                let metrics = OperationMetrics::from_events(op_name, events, total_time_ms);
                (op_name.clone(), metrics)
            })
            .collect();

        let operation_count = operations.len();

        Self {
            metadata: SessionMetadata {
                session_id: uuid::Uuid::new_v4().to_string(),
                start_timestamp: chrono::Utc::now().to_rfc3339(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                context: context_filter.map(String::from),
            },
            operations,
            total_time_ms,
            operation_count,
        }
    }

    /// Convert report to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_event_duration() {
        let event = OperationEvent {
            operation: "test".to_string(),
            backend: "CPU".to_string(),
            data_size: 100,
            start_time_us: 1000,
            end_time_us: 2500,
            context: None,
        };

        assert_eq!(event.duration_us(), 1500);
        assert_eq!(event.duration_ms(), 1.5);
    }

    #[test]
    fn test_operation_metrics_from_events() {
        let events = vec![
            OperationEvent {
                operation: "add".to_string(),
                backend: "CPU".to_string(),
                data_size: 100,
                start_time_us: 0,
                end_time_us: 1000, // 1ms
                context: None,
            },
            OperationEvent {
                operation: "add".to_string(),
                backend: "CPU".to_string(),
                data_size: 200,
                start_time_us: 0,
                end_time_us: 2000, // 2ms
                context: None,
            },
        ];

        let metrics = OperationMetrics::from_events("add", &events, 10.0);

        assert_eq!(metrics.count, 2);
        assert_eq!(metrics.total_time_ms, 3.0);
        assert_eq!(metrics.avg_time_ms, 1.5);
        assert_eq!(metrics.min_time_ms, 1.0);
        assert_eq!(metrics.max_time_ms, 2.0);
        assert_eq!(metrics.primary_backend, "CPU");
        assert_eq!(metrics.percent_total, 30.0); // 3/10 * 100
    }
}
