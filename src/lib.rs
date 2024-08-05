#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

//! [![github]](https://github.com/kands-code/rmatrix)
//! [![crates-io]](https://crates.io/crates/rmatrix_ks)
//! [![docs-rs]](https://docs.rs/rmatrix_ks)
//!
//! [github]: https://img.shields.io/badge/github-8da0cb?style=for-the-badge&labelColor=555555&logo=github
//! [crates-io]: https://img.shields.io/badge/crates.io-fc8d62?style=for-the-badge&labelColor=555555&logo=rust
//! [docs-rs]: https://img.shields.io/badge/docs.rs-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs
//!
//! ## Warning
//!
//! used unstable features
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

pub mod complex;
pub mod error;
pub mod matrix;
pub mod number;
pub mod rational;
pub mod utils;
pub mod vector;

#[cfg(feature = "serde_mat")]
pub mod serde;
