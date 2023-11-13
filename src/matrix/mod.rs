pub mod attr;
pub mod base;
// pub mod json;
pub mod math;
mod shape;
pub mod utils;

use crate::matrix::shape::MatrixShape;

/// length of matrix tag
const TAG_LEANGTH: usize = 8;

#[derive(Debug, Clone)]
/// numeric matrix
pub struct Matrix<N> {
    /// matrix data
    data: Vec<N>,
    /// matrix shape
    shape: MatrixShape,
    /// matrix tag
    pub tag: String,
}
