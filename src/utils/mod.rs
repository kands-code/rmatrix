//! # Utils
//!
//! some useful tools

pub mod common;
pub mod decompose;
pub mod predicate;
pub mod state;

#[cfg(feature = "serde_mat")]
#[doc(cfg(feature = "serde_mat"))]
pub mod serde;
