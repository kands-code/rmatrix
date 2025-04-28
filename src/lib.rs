#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(doc_cfg)]
#![feature(doc_auto_cfg)]
#![feature(generic_const_exprs)]
#![allow(clippy::type_complexity)] // for tuple return

//! # rmatrix_ks
//!
//! [![github]](https://github.com/kands-code/rmatrix)
//! [![crates-io]](https://crates.io/crates/rmatrix_ks)
//! [![docs-rs]](https://docs.rs/rmatrix_ks)
//!
//! [github]: https://img.shields.io/badge/github-8da0cb?style=for-the-badge&labelColor=555555&logo=github
//! [crates-io]: https://img.shields.io/badge/crates.io-fc8d62?style=for-the-badge&labelColor=555555&logo=rust
//! [docs-rs]: https://img.shields.io/badge/docs.rs-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs
//!
//! ## Warnings
//!
//! <div class="warning">
//!
//! Unstable features used:
//! - generic_const_exprs
//!
//! If you want to use this library, please use Rust Nightly
//! and make sure to add `#![feature(generic_const_exprs)]`.
//!
//! </div>
//!
//! # Description
//!
//! This library primarily contains the definitions and implementations
//! of matrix operations and numerical types,
//! and is intended for personal use only.

pub mod matrix;
pub mod number;
