//! # Math
//!
//! Some math functions.

use crate::{matrix::matrix::Matrix, number::traits::number::Number};

pub fn row_eliminate<N: Number, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> (N, Matrix<N, R, R>, Matrix<N, R, R>, Matrix<N, R, C>) {
    todo!()
}
