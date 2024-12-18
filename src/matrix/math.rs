//! # Math
//!
//! Some mathematical functions for matrix operations,
//! such as matrix validation, matrix simplification,
//! and determinant calculation, etc.

use crate::{
    matrix::matrix::Matrix,
    number::{
        instances::complex::Complex,
        traits::{fractional::Fractional, number::Number, realfloat::RealFloat},
    },
};

use super::utils::{conjugate_transpose, points_2d, transpose};

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
///     number::{
///         instances::{double::Double, word8::Word8},
///         traits::zero::Zero,
///     },
/// };
///
/// fn main() {
///     let m1 = Matrix::<Word8, 2, 2>::of(&[Word8::of(1), Word8::of(2), Word8::of(2), Word8::of(1)])
///         .unwrap();
///     let m2 = Matrix::<Word8, 2, 2>::of(&[Word8::of(1), Word8::of(2), Word8::of(1), Word8::of(1)])
///         .unwrap();
///     let m3 = Matrix::<Double, 2, 2>::of(&[
///         Double::of(1.0),
///         Double::of(2.0),
///         Double::of(2.0),
///         Double::of(1.0),
///     ])
///     .unwrap();
///     assert!(is_symmetric_matrix(&m1, |e1, e2| e1 == e2));
///     assert!(!is_symmetric_matrix(&m2, |e1, e2| e1 == e2));
///     assert!(is_symmetric_matrix(&m3, |e1, e2| (e1.clone() - e2.clone()).is_zero()));
/// }
/// ```
pub fn is_symmetric_matrix<N, F, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
    pred: F,
) -> bool
where
    N: PartialEq,
    F: Fn(&N, &N) -> bool,
{
    is_square_matrix(m)
        && points_2d((1, R), (1, C), |row, col| row < col)
            .iter()
            .all(|&p @ (row, col)| pred(&m[p], &m[(col, row)]))
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
    points_2d((1, R), (1, C), |row, col| row > col)
        .iter()
        .all(|&p| m[p].is_zero())
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
    points_2d((1, R), (1, C), |row, col| row < col)
        .iter()
        .all(|&p| m[p].is_zero())
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
    points_2d((1, R), (1, C), |row, col| row != col)
        .iter()
        .cloned()
        .map(|p| &m[p])
        .all(|e| e.is_zero())
}

/// Validate whether a matrix is an identity matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::is_identity_matrix, matrix::Matrix},
///     number::instances::{double::Double, word8::Word8},
/// };
///
/// fn main() {
///     let m1 = Matrix::<Word8, 2, 2>::of(&[Word8::of(1), Word8::of(0), Word8::of(0), Word8::of(1)])
///         .unwrap();
///     let m2 = Matrix::<Double, 2, 2>::of(&[1.0, 2.0, 2.0, 1.0].map(|e| Double::of(e))).unwrap();
///     assert!(is_identity_matrix(&m1));
///     assert!(!is_identity_matrix(&m2));
///     assert!(is_identity_matrix(&Matrix::<Double, 3, 3>::eyes()));
/// }
/// ```
pub fn is_identity_matrix<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> bool
where
    N: Number,
{
    is_square_matrix(m)
        && is_diagonal_matrix(m)
        && points_2d((1, R), (1, C), |row, col| row == col)
            .iter()
            .all(|&p| m[p].is_one())
}

/// Validate whether a matrix is an orthogonal matrix.
///
/// A matrix is called an orthogonal matrix
/// if and only if the transpose of the matrix
/// is the inverse of the matrix itself.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::is_orthogonal_matrix, matrix::Matrix},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m1 = Matrix::<Double, 3, 3>::eyes();
///     let m2 = Matrix::<Double, 2, 2>::of(&[1.0, 2.0, 2.0, 3.0].map(|e| Double::of(e))).unwrap();
///     assert!(is_orthogonal_matrix(&m1));
///     assert!(!is_orthogonal_matrix(&m2));
/// }
/// ```
pub fn is_orthogonal_matrix<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> bool
where
    N: Number,
{
    let transposed = transpose(m);
    if R > C {
        is_identity_matrix(&(transposed * m.clone()))
    } else {
        is_identity_matrix(&(m.clone() * transposed))
    }
}

/// Validate whether a matrix is an unitary matrix.
///
/// A matrix is called an unitary matrix
/// if and only if the conjugate transpose of the matrix
/// is the inverse of the matrix itself.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::is_unitary_matrix, matrix::Matrix},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// fn main() {
///     let m1 = Matrix::<Complex<Float>, 2, 2>::of(&[
///         Complex::of(Float::of(1.0f32 / 2.0f32.sqrt()), Float::zero()),
///         Complex::of(Float::of(-1.0f32 / 2.0f32.sqrt()), Float::zero()),
///         Complex::of(Float::of(1.0f32 / 2.0f32.sqrt()), Float::zero()),
///         Complex::of(Float::of(1.0f32 / 2.0f32.sqrt()), Float::zero()),
///     ])
///     .unwrap();
///     let m2 = Matrix::<Complex<Float>, 2, 2>::of(&[
///         Complex::of(Float::of(1.0f32), Float::zero()),
///         Complex::of(Float::of(-1.0), Float::zero()),
///         Complex::of(Float::zero(), Float::zero()),
///         Complex::of(Float::of(1.0f32), Float::zero()),
///     ])
///     .unwrap();
///     assert!(is_unitary_matrix(&m1));
///     assert!(!is_unitary_matrix(&m2));
/// }
/// ```
pub fn is_unitary_matrix<F, const R: usize, const C: usize>(m: &Matrix<Complex<F>, R, C>) -> bool
where
    F: RealFloat,
{
    let conjugate_transposed = conjugate_transpose(m);
    if R > C {
        is_identity_matrix(&(conjugate_transposed * m.clone()))
    } else {
        is_identity_matrix(&(m.clone() * conjugate_transposed))
    }
}

/// Validate whether a matrix is a hermitian matrix.
///
/// A matrix is called a hermitian matrix
/// if and only if the conjugate transpose of the matrix
/// is the matrix itself.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::is_hermitian_matrix, matrix::Matrix},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// fn main() {
///     let m1 = Matrix::<Complex<Float>, 2, 2>::of(&[
///         Complex::of(Float::of(2.0f32), Float::zero()),
///         Complex::of(Float::of(1.0f32), Float::of(1.0f32)),
///         Complex::of(Float::of(1.0f32), Float::of(-1.0f32)),
///         Complex::of(Float::of(3.0f32), Float::zero()),
///     ])
///     .unwrap();
///     let m2 = Matrix::<Complex<Float>, 2, 2>::of(&[
///         Complex::of(Float::of(1.0f32), Float::zero()),
///         Complex::of(Float::of(2.0f32), Float::of(1.0f32)),
///         Complex::of(Float::of(3.0f32), Float::zero()),
///         Complex::of(Float::of(4.0f32), Float::zero()),
///     ])
///     .unwrap();
///     assert!(is_hermitian_matrix(&m1));
///     assert!(!is_hermitian_matrix(&m2));
/// }
/// ```
pub fn is_hermitian_matrix<F, const R: usize, const C: usize>(m: &Matrix<Complex<F>, R, C>) -> bool
where
    F: RealFloat,
{
    is_square_matrix(m) && m == &conjugate_transpose(m)
}

/// Validate whether a matrix is a normal matrix.
///
/// A matrix `m` is called a normal matrix
/// if and only if it satisfies:
///
/// > mul(m, conj) = mul(conj, m)
///
/// where `conj` is its conjugate transpose.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::is_normal_matrix, matrix::Matrix},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// fn main() {
///     let m1 = Matrix::<Complex<Float>, 2, 2>::of(&[
///         Complex::of(Float::of(1.0f32), Float::zero()),
///         Complex::of(Float::of(2.0f32), Float::zero()),
///         Complex::of(Float::of(2.0f32), Float::zero()),
///         Complex::of(Float::of(3.0f32), Float::zero()),
///     ])
///     .unwrap();
///     let m2 = Matrix::<Complex<Float>, 2, 2>::of(&[
///         Complex::of(Float::of(1.0f32), Float::zero()),
///         Complex::of(Float::of(1.0f32), Float::zero()),
///         Complex::of(Float::of(0.0f32), Float::zero()),
///         Complex::of(Float::of(1.0f32), Float::zero()),
///     ])
///     .unwrap();
///     assert!(is_normal_matrix(&m1));
///     assert!(!is_normal_matrix(&m2));
/// }
/// ```
pub fn is_normal_matrix<F, const R: usize, const C: usize>(m: &Matrix<Complex<F>, R, C>) -> bool
where
    F: RealFloat,
{
    let conjugate_transposed = conjugate_transpose(m);
    is_square_matrix(m)
        && (conjugate_transposed.clone() * m.clone() == m.clone() * conjugate_transposed)
}

/// Calculate the row-reduced form of the matrix.
///
/// # returns
///
/// - Times of row swaps
/// - Row swap matrix
/// - Lower triangular matrix
/// - Row-reduced matrix
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::row_reduce, matrix::Matrix},
///     number::{instances::double::Double, traits::zero::Zero},
/// };
///
/// fn main() {
///     let m = Matrix::<Double, 3, 3>::of(
///         &[
///             2.0, 1.0, -1.0, // r1
///             -3.0, -1.0, 2.0, // r2
///             -2.0, 1.0, 2.0, // r3
///         ]
///         .map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let (_, _, _, reduced) = row_reduce(&m);
///     assert!((reduced
///         - Matrix::<Double, 3, 3>::of(
///             &[
///                 2.0, 1.0, -1.0, //r1
///                 0.0, 0.5, 0.5, // r2
///                 0.0, 0.0, -1.0 // r3
///             ]
///             .map(|e| Double::of(e))
///         )
///         .unwrap())
///     .linear_iter()
///     .all(|e| e.is_zero()));
/// }
/// ```
pub fn row_reduce<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> (usize, Matrix<N, R, R>, Matrix<N, R, R>, Matrix<N, R, C>)
where
    N: Fractional,
{
    let mut t = 0;
    let mut p = Matrix::<N, R, R>::eyes();
    let mut lt = Matrix::<N, R, R>::eyes();
    let mut reduced = m.clone();

    if !(is_upper_triangular_matrix(&reduced) || R < 2) {
        // Deviation of the pivot.
        let mut deviation = 0;
        for row in 1..R {
            // Boundary check.
            if row + deviation > C {
                break;
            }
            // Pivot check.
            let mut pivot_check = reduced[(row, row + deviation)].is_zero();
            while pivot_check && (row + deviation) <= C {
                // Find the row where the pivot at the corresponding position is non-zero.
                for next in (row + 1)..=R {
                    // Perform row swapping.
                    if !reduced[(next, row + deviation)].is_zero() {
                        let swap = Matrix::<N, R, R>::p_change(row, next);
                        p = swap.clone() * p;
                        reduced = swap * reduced;
                        t = t + 1;
                        pivot_check = false;
                    }
                }
                // Check the pivot of the next column.
                if pivot_check {
                    deviation = deviation + 1;
                } else {
                    break;
                }
            }
            if pivot_check {
                // No suitable row for swapping, exiting.
                break;
            } else {
                // Perform row reduce.
                let pivot = reduced[(row, row + deviation)].clone();
                for next in (row + 1)..=R {
                    let next_pivot = reduced[(next, row + deviation)].clone();
                    if !next_pivot.is_zero() {
                        let factor = next_pivot / pivot.clone();
                        let add = Matrix::<N, R, R>::p_add(row, next, -factor);
                        lt = add.clone() * lt;
                        reduced = add * reduced;
                    }
                }
            }
        }
    }
    // returns
    (t, p, lt, reduced)
}

/// Calculate the rank of the matrix.
///
/// # Examples
///
/// ```rust
///
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
                    "Error[matrix::rank]: get the {}-th row of matrix failed",
                    row_index
                ))
                .linear_iter()
                .any(|e| !e.is_zero())
        })
        .filter(|&p| p)
        .count()
}
