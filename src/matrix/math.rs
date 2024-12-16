//! # Math
//!
//! Some mathematical functions for matrix operations,
//! such as matrix validation, matrix simplification,
//! and determinant calculation, etc.

use crate::{matrix::matrix::Matrix, number::traits::number::Number};

use super::utils::points_2d;

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
        && points_2d((1, R), (1, C), Some(|x, y| x < y))
            .iter()
            .cloned()
            .map(|(x, y)| (&m[(x, y)], &m[(y, x)]))
            .all(|(e1, e2)| e1 == e2)
}

/// Validate whether a matrix is an upper triangular matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::is_upper_triangular_matrix, matrix::Matrix},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m1: Matrix<Word8, 2, 2> =
///         Matrix::of(&[Word8::of(1), Word8::of(1), Word8::of(0), Word8::of(1)]).unwrap();
///     assert!(is_upper_triangular_matrix(&m1));
///
///     let m2: Matrix<Word8, 2, 2> =
///         Matrix::of(&[Word8::of(1), Word8::of(0), Word8::of(1), Word8::of(1)]).unwrap();
///     assert!(!is_upper_triangular_matrix(&m2));
/// }
/// ```
pub fn is_upper_triangular_matrix<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> bool
where
    N: Number,
{
    points_2d((1, R), (1, C), Some(|x, y| x > y))
        .iter()
        .cloned()
        .map(|(x, y)| &m[(x, y)])
        .all(|e| e.is_zero())
}

/// Validate whether a matrix is a lower triangular matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::is_lower_triangular_matrix, matrix::Matrix},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m1: Matrix<Word8, 2, 2> =
///         Matrix::of(&[Word8::of(1), Word8::of(1), Word8::of(0), Word8::of(1)]).unwrap();
///     assert!(!is_lower_triangular_matrix(&m1));
///
///     let m2: Matrix<Word8, 2, 2> =
///         Matrix::of(&[Word8::of(1), Word8::of(0), Word8::of(1), Word8::of(1)]).unwrap();
///     assert!(is_lower_triangular_matrix(&m2));
/// }
/// ```
pub fn is_lower_triangular_matrix<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> bool
where
    N: Number,
{
    points_2d((1, R), (1, C), Some(|x, y| x < y))
        .iter()
        .cloned()
        .map(|(x, y)| &m[(x, y)])
        .all(|e| e.is_zero())
}

/// Validate whether a matrix is a diagonal matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::is_diagonal_matrix, matrix::Matrix},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m1: Matrix<Word8, 2, 2> =
///         Matrix::of(&[Word8::of(1), Word8::of(0), Word8::of(0), Word8::of(2)]).unwrap();
///     assert!(is_diagonal_matrix(&m1));
///
///     let m2: Matrix<Word8, 2, 2> =
///         Matrix::of(&[Word8::of(1), Word8::of(0), Word8::of(1), Word8::of(1)]).unwrap();
///     assert!(!is_diagonal_matrix(&m2));
/// }
/// ```
pub fn is_diagonal_matrix<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> bool
where
    N: Number,
{
    points_2d((1, R), (1, C), Some(|x, y| x != y))
        .iter()
        .cloned()
        .map(|(x, y)| &m[(x, y)])
        .all(|e| e.is_zero())
}
