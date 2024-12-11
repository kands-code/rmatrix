//! # Vector
//!
//! Row vector and column vector definations and operations.

use crate::{matrix::matrix::Matrix, number::traits::number::Number};

pub type VectorR<N, const C: usize> = Matrix<N, 1, C>;
pub type VectorC<N, const R: usize> = Matrix<N, R, 1>;

pub fn index_c<N, const R: usize>(v: &VectorC<N, R>, row_index: usize) -> &N {
    &v[(row_index, 1)]
}

pub fn index_r<N, const C: usize>(v: &VectorR<N, C>, column_index: usize) -> &N {
    &v[(1, column_index)]
}

pub fn dot_product<N, const R: usize>(v1: VectorC<N, R>, v2: VectorC<N, R>) -> N
where
    N: Number,
{
    v1.linear_iter()
        .zip(v2.linear_iter())
        .map(|(e1, e2)| e1.clone() * e2.clone())
        .fold(N::zero(), |acc, e| acc + e)
}

pub fn cross_product<N>(v1: VectorC<N, 3>, v2: VectorC<N, 3>) -> VectorC<N, 3>
where
    N: Number,
{
    let mut inner = vec![N::zero(); 3];
    inner[0] = index_c(&v1, 2).clone() * index_c(&v2, 3).clone()
        - index_c(&v1, 3).clone() * index_c(&v2, 2).clone();
    inner[1] = index_c(&v1, 3).clone() * index_c(&v2, 1).clone()
        - index_c(&v1, 1).clone() * index_c(&v2, 3).clone();
    inner[2] = index_c(&v1, 1).clone() * index_c(&v2, 2).clone()
        - index_c(&v1, 2).clone() * index_c(&v2, 1).clone();

    VectorC { inner }
}

pub fn layer_product<N, const R: usize, const C: usize>(
    v1: VectorC<N, R>,
    v2: VectorR<N, C>,
) -> Matrix<N, R, C>
where
    N: Number,
{
    let mut inner = Vec::with_capacity(R * C);
    for row_index in 1..=R {
        for column_index in 1..=C {
            inner.push(index_c(&v1, row_index).clone() * index_r(&v2, column_index).clone());
        }
    }
    Matrix { inner }
}
