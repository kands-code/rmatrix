//! type of complex number

pub mod base;
pub mod num;
pub mod utils;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex(f64, f64);
