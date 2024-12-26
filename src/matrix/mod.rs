//! # matrix
//!
//! Matrix and relative operations.
//!
//! > All matrix indices are 1-based.

pub mod math;
pub mod matrix;
pub mod serde;
pub mod utils;
pub mod vector;

#[cfg(feature = "complex")]
pub mod complex;
#[cfg(feature = "complex")]
pub mod extra;
