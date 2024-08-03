#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

//! ## Warning
//!
//! used incomplete features
//! - `generic_const_exprs`
//!
//! # Description
//!
//! This library is for matrix computation,
//! just for learning and self using.
//!
//! There are some optional features
//! - rand_mat: randomly generated matrices
//! - serde_mat: ability to read matrices using a specific file format
//! - rayon_mat: use rayon iter instead of std::iter
//!
//! By default, `rand_mat` feature is enabled.

pub mod error;
pub mod matrix;
pub mod number;
pub mod rational;
pub mod utils;
pub mod vector;

#[cfg(feature = "serde_mat")]
pub mod serde;
