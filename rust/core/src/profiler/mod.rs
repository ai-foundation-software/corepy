//! Profiler module for corepy
//!
//! This module implements the performance profiling system.

pub mod core;
pub mod metrics;

pub use self::core::{set_context, ProfileScope, Profiler};
