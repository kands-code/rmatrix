//! # matrix
//!
//! Matrix and relative operations.
//!
//! > All matrix indices are 1-based.

/// Default maximum number of iterations.
///
/// The default iteration limit for all functions
/// that use iterative methods for solving internally.
pub const DEFAULT_MAX_ITER: usize = 1024;

pub mod complex;
pub mod extra;
pub mod math;
pub mod matrix;
pub mod serde;
pub mod utils;
pub mod vector;
