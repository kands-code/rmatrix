pub mod attr;
pub mod base;
pub mod math;
pub mod normal;
mod shape;

use crate::matrix::shape::MatrixShape;
#[cfg(feature = "serialize")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
/// normal matrix with data and size
pub struct Matrix {
    /// data for matrix
    data: Vec<f64>,
    /// size for matrix
    shape: MatrixShape,
    /// tag for matrix
    pub tag: String,
}
