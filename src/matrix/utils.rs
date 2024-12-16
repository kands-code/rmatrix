//! # Utils
//!
//! Some util functions.

use crate::{
    matrix::matrix::Matrix,
    number::{
        instances::complex::Complex,
        traits::{number::Number, realfloat::RealFloat},
    },
};

pub fn points_2d<F>(
    (x_lb, x_ub): (usize, usize),
    (y_lb, y_ub): (usize, usize),
    predicate: Option<F>,
) -> Vec<(usize, usize)>
where
    F: Fn(usize, usize) -> bool,
{
    if x_lb >= x_ub || y_lb >= y_ub {
        Vec::new()
    } else {
        let mut all_points = Vec::with_capacity((x_ub - x_lb) * (y_ub - y_lb));
        for x in x_lb..=x_ub {
            for y in y_lb..=y_ub {
                if if let Some(ref p) = predicate {
                    p(x, y)
                } else {
                    true
                } {
                    all_points.push((x, y));
                }
            }
        }
        all_points
    }
}

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

pub fn transpose<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> Matrix<N, C, R>
where
    N: Clone,
{
    let mut inner = Vec::with_capacity(R * C);
    for row_index in 1..=C {
        for column_index in 1..=R {
            inner.push(m[(column_index, row_index)].clone());
        }
    }
    Matrix { inner }
}

pub fn conjugate_transpose<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> Matrix<Complex<F>, C, R>
where
    F: RealFloat,
{
    let transposed = transpose(m);
    map(&transposed, |e| e.conjugate())
}

pub fn horizontal_concat<N, const R: usize, const C1: usize, const C2: usize>(
    m1: &Matrix<N, R, C1>,
    m2: &Matrix<N, R, C2>,
) -> Matrix<N, R, { C1 + C2 }>
where
    N: Clone,
{
    let mut inner = Vec::with_capacity(R * (C1 + C2));
    for row_index in 1..=R {
        for column_index in 1..=(C1 + C2) {
            inner.push(if column_index <= C1 {
                m1[(row_index, column_index)].clone()
            } else {
                m2[(row_index, column_index - C1)].clone()
            });
        }
    }
    Matrix { inner }
}

pub fn vertical_concat<N, const R1: usize, const R2: usize, const C: usize>(
    m1: &Matrix<N, R1, C>,
    m2: &Matrix<N, R2, C>,
) -> Matrix<N, { R1 + R2 }, C>
where
    N: Clone,
{
    let mut inner = Vec::with_capacity((R1 + R2) * C);
    for row_index in 1..=(R1 + R2) {
        for column_index in 1..=C {
            inner.push(if row_index <= R1 {
                m1[(row_index, column_index)].clone()
            } else {
                m2[(row_index - R1, column_index)].clone()
            });
        }
    }
    Matrix { inner }
}
