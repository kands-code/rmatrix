#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

//! *warn*: used incomplete features
//! - `generic_const_exprs`
//!
//! # Description
//!
//! This library is for matrix computation,
//! just for learning and self using.
//!
//! There are two optional features, `rand_mat` and `serde_mat`,
//! corresponding to randomly generated matrices
//! and the ability to read matrices using a specific file format.
//! By default, both features are enabled.

pub mod error;
pub mod matrix;
pub mod number;
pub mod rational;
pub mod utils;
pub mod vector;

#[cfg(feature = "serde_mat")]
pub mod serde;
