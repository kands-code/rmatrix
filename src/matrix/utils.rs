//! # matrix::utils
//!
//! Some util functions.

use crate::{
    matrix::{math::row_eliminate, matrix::Matrix},
    number::{
        instances::complex::Complex,
        traits::{fractional::Fractional, number::Number, realfloat::RealFloat},
    },
};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::{math::row_reduce, vector::VectorC};

/// Generates coordinates within a specified inclusive-range that meet certain criteria.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::matrix::utils::points_2d;
///
/// fn main() {
///     // Generates coordinates for the lower triangular part of a 3x3 matrix.
///     let p = points_2d((1, 3), (1, 3), |row, column| row > column);
///     assert_eq!(p, vec![(2, 1), (3, 1), (3, 2)]);
/// }
/// ```
pub fn points_2d<F>(
    (row_lb, row_ub): (usize, usize),
    (col_lb, col_ub): (usize, usize),
    criteria: F,
) -> Vec<(usize, usize)>
where
    F: Fn(usize, usize) -> bool,
{
    if row_lb > row_ub || col_lb > col_ub {
        Vec::new()
    } else {
        let mut all_points = Vec::with_capacity((row_ub - row_lb + 1) * (col_ub - col_lb + 1));
        for row in row_lb..=row_ub {
            for col in col_lb..=col_ub {
                if criteria(row, col) {
                    all_points.push((row, col));
                }
            }
        }
        // used to save a certain amount of space
        all_points.shrink_to_fit();
        all_points
    }
}

/// Calculates the trace of the matrix.
///
/// # Panics
///
/// This function requires the use of the `#![feature(generic_const_exprs)]`.
///
/// # Examples
///
/// ```rust
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::trace},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m = Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
///     assert_eq!(trace(&m), Word8::of(15));
/// }
/// ```
pub fn trace<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> N
where
    N: Number,
    [(); Matrix::<N, R, C>::get_diagonal_length()]:,
{
    m.get_diagonal()
        .linear_iter()
        .cloned()
        .fold(N::zero(), |acc, e| acc + e.clone())
}

/// Applies the function f to each element of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::apply},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m = Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
///     let n =
///         Matrix::<Word8, 3, 3>::of(&[2, 4, 6, 8, 10, 12, 14, 16, 18].map(|e| Word8::of(e))).unwrap();
///     assert_eq!(apply(&m, |e| e * Word8::of(2)), n);
/// }
/// ```
pub fn apply<N, M, F, const R: usize, const C: usize>(m: &Matrix<N, R, C>, f: F) -> Matrix<M, R, C>
where
    N: Sync + Clone,
    M: Send + Sync,
    F: Fn(N) -> M + Sync,
{
    let inner = m.inner.par_iter().map(|e| f(e.clone())).collect();
    Matrix { inner }
}

/// Obtains the transpose of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::transpose},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m = Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
///     let n = Matrix::<Word8, 3, 3>::of(&[1, 4, 7, 2, 5, 8, 3, 6, 9].map(|e| Word8::of(e))).unwrap();
///     assert_eq!(transpose(&m), n);
/// }
/// ```
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

/// Obtains the conjugate transpose of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::conjugate_transpose},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 2, 2>::of(
///         &[(1.0, 2.0), (3.0, -1.0), (4.0, 0.0), (5.0, 6.0)]
///             .map(|(real, imag)| Complex::of(Float::of(real), Float::of(imag))),
///     )
///     .unwrap();
///     let n = Matrix::<Complex<Float>, 2, 2>::of(
///         &[(1.0, -2.0), (4.0, 0.0), (3.0, 1.0), (5.0, -6.0)]
///             .map(|(real, imag)| Complex::of(Float::of(real), Float::of(imag))),
///     )
///     .unwrap();
///     assert!(conjugate_transpose(&m).equals(&n));
/// }
/// ```
pub fn conjugate_transpose<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> Matrix<Complex<F>, C, R>
where
    F: RealFloat,
{
    let transposed = transpose(m);
    apply(&transposed, |e| e.conjugate())
}

/// Calculate the rank of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{utils::rank, matrix::Matrix},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m = Matrix::<Double, 3, 3>::of(
///         &[2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0].map(|e| Double::of(e)),
///     )
///     .unwrap();
///     assert_eq!(rank(&m), 3);
/// }
/// ```
pub fn rank<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> usize
where
    N: Fractional,
{
    let (_, _, _, reduced) = row_reduce(m);
    (1..=R)
        .map(|row_index| {
            reduced
                .get_row(row_index)
                .expect(&format!(
                    "Error[matrix::utils::rank]: Failed to retrieve the {}-th row of the matrix.",
                    row_index
                ))
                .inner
                .par_iter()
                .any(|e| !e.is_zero())
        })
        .filter(|&p| p)
        .count()
}

/// Calculate the nullspace of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::nullspace, vector::VectorC},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let a = Matrix::<Double, 3, 5>::of(
///         &[
///             -3.0, 6.0, -1.0, 1.0, -7.0, 1.0, -2.0, 2.0, 3.0, -1.0, 2.0, -4.0, 5.0, 8.0, -4.0,
///         ]
///         .map(Double::of),
///     )
///     .unwrap();
///     let ns = nullspace(&a);
///     // For this matrix, the nullspace contains only three elements.
///     assert_eq!(ns.len(), 3);
///     // Second column:
///     let n1 = VectorC::<Double, 5>::of(&[2.0, 1.0, 0.0, 0.0, 0.0].map(Double::of)).unwrap();
///     assert_eq!(ns[0], n1);
///     // Fourth column:
///     let n2 = VectorC::<Double, 5>::of(&[1.0, 0.0, -2.0, 1.0, 0.0].map(Double::of)).unwrap();
///     assert_eq!(ns[1], n2);
///     // Fifth column:
///     let n3 = VectorC::<Double, 5>::of(&[-3.0, 0.0, 2.0, 0.0, 1.0].map(Double::of)).unwrap();
///     assert_eq!(ns[2], n3);
/// }
/// ```
pub fn nullspace<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> Vec<VectorC<N, C>>
where
    N: Fractional,
{
    // Reduce the matrix to its row echelon form.
    let refined = row_eliminate(m);
    // Set the markers for each row, which is the column index of the first non-zero element.
    let mut row_flags: [usize; C] = [0; C];
    for row in 1..=R {
        for column in row..=C {
            if !(row_flags.contains(&row) || refined[(row, column)].is_zero()) {
                row_flags[column - 1] = row;
            }
        }
    }
    let mut space = Vec::new();
    // Only rank-deficient matrices have a nullspace.
    if row_flags.iter().any(|&e| e == 0) {
        for column in 1..=C {
            // The vector corresponding to unmarked columns is an element of the nullspace.
            if row_flags[column - 1] == 0 {
                let mut base = VectorC::<N, C>::default();
                for check in 1..=C {
                    if row_flags[check - 1] != 0 {
                        // Negate the elements of the marked rows.
                        let val = refined[(row_flags[check - 1], column)].clone();
                        base[(check, 1)] = if val.is_zero() { val } else { -val };
                    } else if column == check {
                        // Set the column corresponding to itself to one.
                        base[(check, 1)] = N::one();
                    }
                }
                space.push(base);
            }
        }
    }
    space
}

/// Horizontally concatenate two matrices.
///
/// # Panics
///
/// This function requires the use of the `#![feature(generic_const_exprs)]`.
///
/// # Examples
///
/// ```rust
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::horizontal_concat},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m = Matrix::<Word8, 2, 2>::of(&[1, 2, 3, 4].map(|e| Word8::of(e))).unwrap();
///     let n = Matrix::<Word8, 2, 2>::of(&[5, 6, 7, 8].map(|e| Word8::of(e))).unwrap();
///     let cat = Matrix::<Word8, 2, 4>::of(&[1, 2, 5, 6, 3, 4, 7, 8].map(|e| Word8::of(e))).unwrap();
///     assert_eq!(horizontal_concat(&m, &n), cat);
/// }
/// ```
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

/// Vertically concatenate two matrices.
///
/// # Panics
///
/// This function requires the use of the `#![feature(generic_const_exprs)]`.
///
/// # Examples
///
/// ```rust
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::vertical_concat},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m = Matrix::<Word8, 2, 2>::of(&[1, 2, 3, 4].map(|e| Word8::of(e))).unwrap();
///     let n = Matrix::<Word8, 2, 2>::of(&[5, 6, 7, 8].map(|e| Word8::of(e))).unwrap();
///     let cat = Matrix::<Word8, 4, 2>::of(&[1, 2, 3, 4, 5, 6, 7, 8].map(|e| Word8::of(e))).unwrap();
///     assert_eq!(vertical_concat(&m, &n), cat);
/// }
/// ```
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
