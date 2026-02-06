//! Thread-safe global profiler
//!
//! Provides a global profiler instance that can be safely accessed from
//! multiple threads. Profiling is disabled by default and has zero overhead
//! when disabled.

use super::metrics::{OperationEvent, ProfileReport};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Thread-safe global profiler state
#[derive(Clone)]
pub struct Profiler {
    /// Whether profiling is currently enabled
    enabled: Arc<RwLock<bool>>,

    /// Collected profiling events
    events: Arc<RwLock<Vec<OperationEvent>>>,
}

impl Profiler {
    /// Create a new profiler (disabled by default)
    pub fn new() -> Self {
        Self {
            enabled: Arc::new(RwLock::new(false)),
            events: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Enable profiling
    pub fn enable(&self) {
        *self.enabled.write() = true;
    }

    /// Disable profiling
    pub fn disable(&self) {
        *self.enabled.write() = false;
    }

    /// Check if profiling is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        *self.enabled.read()
    }

    /// Record an operation event
    ///
    /// This is a hot path function - optimized for minimal overhead
    pub fn record_operation(
        &self,
        operation: String,
        backend: String,
        data_size: usize,
        start_time_us: u64,
        end_time_us: u64,
        context: Option<String>,
    ) {
        // Fast path: if profiling is disabled, return immediately
        if !self.is_enabled() {
            return;
        }

        let event = OperationEvent {
            operation,
            backend,
            data_size,
            start_time_us,
            end_time_us,
            context,
        };

        self.events.write().push(event);
    }

    /// Clear all recorded events
    pub fn clear(&self) {
        self.events.write().clear();
    }

    /// Get the number of recorded events
    #[allow(dead_code)]
    pub fn event_count(&self) -> usize {
        self.events.read().len()
    }

    /// Generate a profiling report
    pub fn generate_report(&self, context_filter: Option<&str>) -> ProfileReport {
        let events = self.events.read();
        ProfileReport::from_events(&events, context_filter)
    }

    /// Export events as JSON
    pub fn export_json(&self, context_filter: Option<&str>) -> Result<String, String> {
        let report = self.generate_report(context_filter);
        report
            .to_json()
            .map_err(|e| format!("JSON serialization failed: {}", e))
    }

    /// Get all events (for advanced use cases)
    #[allow(dead_code)]
    pub fn get_events(&self) -> Vec<OperationEvent> {
        self.events.read().clone()
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local profiler instance for zero overhead when disabled
thread_local! {
    static PROFILER_CONTEXT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// Get current timestamp in microseconds
#[inline]
pub fn now_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_micros() as u64
}

/// RAII guard for profiling a scope
///
/// Automatically records the operation when dropped
pub struct ProfileScope {
    profiler: Profiler,
    operation: String,
    backend: String,
    data_size: usize,
    start_time_us: u64,
    context: Option<String>,
}

impl ProfileScope {
    /// Create a new profile scope
    pub fn new(profiler: Profiler, operation: String, backend: String, data_size: usize) -> Self {
        let context =
            PROFILER_CONTEXT.with(|ctx: &std::cell::RefCell<Option<String>>| ctx.borrow().clone());

        Self {
            profiler,
            operation,
            backend,
            data_size,
            start_time_us: now_micros(),
            context,
        }
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        let end_time_us = now_micros();

        self.profiler.record_operation(
            self.operation.clone(),
            self.backend.clone(),
            self.data_size,
            self.start_time_us,
            end_time_us,
            self.context.clone(),
        );
    }
}

/// Set the current profiling context
pub fn set_context(context: Option<String>) {
    PROFILER_CONTEXT.with(|ctx: &std::cell::RefCell<Option<String>>| {
        *ctx.borrow_mut() = context;
    });
}

/// Get the current profiling context
#[allow(dead_code)]
pub fn get_context() -> Option<String> {
    PROFILER_CONTEXT.with(|ctx: &std::cell::RefCell<Option<String>>| ctx.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_profiler_enable_disable() {
        let profiler = Profiler::new();
        assert!(!profiler.is_enabled());

        profiler.enable();
        assert!(profiler.is_enabled());

        profiler.disable();
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_record_operation() {
        let profiler = Profiler::new();
        profiler.enable();

        profiler.record_operation(
            "test_op".to_string(),
            "CPU".to_string(),
            100,
            1000,
            2000,
            None,
        );

        assert_eq!(profiler.event_count(), 1);

        let events = profiler.get_events();
        assert_eq!(events[0].operation, "test_op");
        assert_eq!(events[0].duration_us(), 1000);
    }

    #[test]
    fn test_profiler_when_disabled() {
        let profiler = Profiler::new();
        // Profiling disabled by default

        profiler.record_operation(
            "test_op".to_string(),
            "CPU".to_string(),
            100,
            1000,
            2000,
            None,
        );

        // Should not record when disabled
        assert_eq!(profiler.event_count(), 0);
    }

    #[test]
    fn test_clear() {
        let profiler = Profiler::new();
        profiler.enable();

        profiler.record_operation(
            "test_op".to_string(),
            "CPU".to_string(),
            100,
            1000,
            2000,
            None,
        );

        assert_eq!(profiler.event_count(), 1);

        profiler.clear();
        assert_eq!(profiler.event_count(), 0);
    }

    #[test]
    fn test_profile_scope() {
        let profiler = Profiler::new();
        profiler.enable();

        {
            let _scope = ProfileScope::new(
                profiler.clone(),
                "scoped_op".to_string(),
                "CPU".to_string(),
                100,
            );

            // Simulate some work
            thread::sleep(Duration::from_millis(1));
        } // Scope ends here, operation is recorded

        assert_eq!(profiler.event_count(), 1);
        let events = profiler.get_events();
        assert_eq!(events[0].operation, "scoped_op");
        assert!(events[0].duration_us() >= 1000); // At least 1ms
    }

    #[test]
    fn test_context_tracking() {
        set_context(Some("test_context".to_string()));
        assert_eq!(get_context(), Some("test_context".to_string()));

        set_context(None);
        assert_eq!(get_context(), None);
    }

    #[test]
    fn test_generate_report() {
        let profiler = Profiler::new();
        profiler.enable();

        // Record multiple operations
        profiler.record_operation("add".to_string(), "CPU".to_string(), 100, 0, 1000, None);

        profiler.record_operation("add".to_string(), "CPU".to_string(), 200, 0, 2000, None);

        profiler.record_operation("mul".to_string(), "GPU".to_string(), 100, 0, 500, None);

        let report = profiler.generate_report(None);

        assert_eq!(report.operation_count, 2); // "add" and "mul"
        assert_eq!(report.total_time_ms, 3.5); // 1 + 2 + 0.5 ms

        let add_metrics = report.operations.get("add").unwrap();
        assert_eq!(add_metrics.count, 2);
        assert_eq!(add_metrics.total_time_ms, 3.0);

        let mul_metrics = report.operations.get("mul").unwrap();
        assert_eq!(mul_metrics.count, 1);
        assert_eq!(mul_metrics.total_time_ms, 0.5);
    }
}
