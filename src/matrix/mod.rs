pub mod attr;
pub mod base;
pub mod math;
pub mod normal;
mod shape;

use crate::matrix::shape::MatrixShape;
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
