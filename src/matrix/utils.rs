//! # Utils
//!
//! Some util functions.

use crate::{matrix::matrix::Matrix, number::traits::number::Number};

pub fn trace<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> N
where
    N: Number,
    [(); Matrix::<N, R, C>::get_diagonal_length()]:,
{
    m.get_diagonal()
        .linear_iter()
        .cloned()
        .fold(N::default(), |acc, e| acc + e.clone())
}

pub fn map<N, M, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
    f: impl Fn(N) -> M,
) -> Matrix<M, R, C>
where
    N: Clone,
{
    let inner = m.linear_iter().map(|e| f(e.clone())).collect();
    Matrix { inner }
}
