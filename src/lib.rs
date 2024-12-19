#![allow(incomplete_features)]
#![feature(doc_cfg)]
#![feature(generic_const_exprs)]

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
//! ## Warning
//!
//! Unstable features used:
//! - generic_const_exprs
//!
//! Please add `#![feature(generic_const_exprs)]` if you want to use this library.
//!
//! # Description
//!
//! This library is used for numerical computing,
//! mainly matrix operations as well as numerical ordinary differential equations
//! and some optimization problems, just for learning and self using.

pub mod matrix;
pub mod number;
