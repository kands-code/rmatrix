//! # Math
//!
//! Some mathematical functions for matrix operations,
//! such as matrix validation, matrix simplification,
//! and determinant calculation, etc.

use crate::matrix::matrix::Matrix;

/// Validate whether a matrix is a square matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::is_square_matrix, matrix::Matrix},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m1 = Matrix::<Word8, 2, 2>::of(&[Word8::of(1), Word8::of(2), Word8::of(2), Word8::of(1)])
///         .unwrap();
///     let m2 = Matrix::<Word8, 2, 1>::of(&[Word8::of(1), Word8::of(3)]).unwrap();
///     assert!(is_square_matrix(&m1));
///     assert!(!is_square_matrix(&m2));
/// }
/// ```
pub const fn is_square_matrix<N, const R: usize, const C: usize>(_: &Matrix<N, R, C>) -> bool {
    R == C
}

/// Validate whether a matrix is a symmetric matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::is_symmetric_matrix, matrix::Matrix},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m1 = Matrix::<Word8, 2, 2>::of(&[Word8::of(1), Word8::of(2), Word8::of(2), Word8::of(1)])
///         .unwrap();
///     let m2 = Matrix::<Word8, 2, 2>::of(&[Word8::of(1), Word8::of(2), Word8::of(1), Word8::of(1)])
///         .unwrap();
///     assert!(is_symmetric_matrix(&m1));
///     assert!(!is_symmetric_matrix(&m2));
/// }
/// ```
pub fn is_symmetric_matrix<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> bool
where
    N: PartialEq,
{
    is_square_matrix(m)
        && (2..=R)
            .map(|row| {
                (1..row)
                    .map(|column| (&m[(row, column)], &m[(column, row)]))
                    .collect::<Vec<(&N, &N)>>()
            })
            .flatten()
            .all(|(e1, e2)| e1 == e2)
}
