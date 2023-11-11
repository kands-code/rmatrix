pub mod matrix_base;
pub mod matrix_fn;
pub mod matrix_ops;
mod matrix_shape;

use crate::core::matrix_shape::MatrixShape;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// normal matrix with data and size
pub struct Matrix {
    /// data for matrix
    data: Vec<f64>,
    /// size for matrix
    shape: MatrixShape,
    /// tag for matrix
    pub tag: String,
}
