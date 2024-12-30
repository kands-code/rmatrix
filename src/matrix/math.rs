//! # matrix::math
//!
//! Some mathematical functions for matrix operations,
//! such as matrix validation, matrix simplification,
//! and determinant calculation, etc.

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    matrix::{
        matrix::Matrix,
        utils::{points_2d, transpose},
    },
    number::{
        traits::{floating::Floating, fractional::Fractional, number::Number, real::Real},
        utils::{inversion_count, permutation},
    },
};

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
///     let m1 = Matrix::<Word8, 2, 2>::of(&[
///         Word8::of(1),
///         Word8::of(2),
///         Word8::of(2),
///         Word8::of(1),
///     ])
///     .unwrap();
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
///     let m1 = Matrix::<Word8, 2, 2>::of(&[
///         Word8::of(1),
///         Word8::of(2),
///         Word8::of(2),
///         Word8::of(1),
///     ])
///     .unwrap();
///     let m2 = Matrix::<Word8, 2, 2>::of(&[
///         Word8::of(1),
///         Word8::of(2),
///         Word8::of(1),
///         Word8::of(1),
///     ])
///     .unwrap();
///     let m3 = Matrix::<Double, 2, 2>::of(&[
///         Double::of(1.0),
///         Double::of(2.0),
///         Double::of(2.0),
///         Double::of(1.0),
///     ])
///     .unwrap();
///     assert!(is_symmetric_matrix(&m1));
///     assert!(!is_symmetric_matrix(&m2));
///     assert!(is_symmetric_matrix(&m3));
/// }
/// ```
pub fn is_symmetric_matrix<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> bool
where
    N: PartialEq + Sync,
{
    is_square_matrix(m)
        && points_2d((1, R), (1, C), |row, col| row < col)
            .par_iter()
            .all(|&p @ (row, col)| &m[p] == &m[(col, row)])
}

/// Validate whether a matrix is an anti-symmetric matrix.
///
/// trans(A) = -A
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::is_anti_symmetric_matrix, matrix::Matrix},
///     number::{
///         instances::{double::Double, int8::Int8},
///         traits::zero::Zero,
///     },
/// };
///
/// fn main() {
///     let m1 =
///         Matrix::<Int8, 2, 2>::of(&[Int8::of(0), Int8::of(2), Int8::of(-2), Int8::of(0)])
///             .unwrap();
///     let m2 =
///         Matrix::<Int8, 2, 2>::of(&[Int8::of(1), Int8::of(2), Int8::of(-2), Int8::of(1)])
///             .unwrap();
///     let m3 = Matrix::<Double, 2, 2>::of(&[
///         Double::of(0.0),
///         Double::of(0.0),
///         Double::of(0.0),
///         Double::of(0.0),
///     ])
///     .unwrap();
///     assert!(is_anti_symmetric_matrix(&m1));
///     assert!(!is_anti_symmetric_matrix(&m2));
///     assert!(is_anti_symmetric_matrix(&m3));
/// }
/// ```
pub fn is_anti_symmetric_matrix<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> bool
where
    N: Real,
{
    is_square_matrix(m)
        && points_2d((1, R), (1, C), |row, col| row <= col)
            .par_iter()
            .all(|&p @ (row, col)| m[p] == -m[(col, row)].clone())
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
pub fn is_upper_triangular_matrix<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> bool
where
    N: Number,
{
    points_2d((1, R), (1, C), |row, col| row > col)
        .par_iter()
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
pub fn is_lower_triangular_matrix<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> bool
where
    N: Number,
{
    points_2d((1, R), (1, C), |row, col| row < col)
        .par_iter()
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
        .par_iter()
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
///     let m1 = Matrix::<Word8, 2, 2>::of(&[
///         Word8::of(1),
///         Word8::of(0),
///         Word8::of(0),
///         Word8::of(1),
///     ])
///     .unwrap();
///     let m2 =
///         Matrix::<Double, 2, 2>::of(&[1.0, 2.0, 2.0, 1.0].map(|e| Double::of(e))).unwrap();
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
            .par_iter()
            .all(|&p| m[p].is_one())
}

/// Validate whether a matrix is an normal matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::is_normal_matrix, matrix::Matrix},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let m = Matrix::<Float, 2, 2>::of(&[1.0, 2.0, -2.0, 1.0].map(Float::of)).unwrap();
///     assert!(is_normal_matrix(&m));
/// }
/// ```
pub fn is_normal_matrix<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> bool
where
    N: Real,
{
    let transposed = transpose(m);
    is_square_matrix(m) && (transposed.clone() * m.clone() == m.clone() * transposed)
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
///     let m2 =
///         Matrix::<Double, 2, 2>::of(&[1.0, 2.0, 2.0, 3.0].map(|e| Double::of(e))).unwrap();
///     assert!(is_orthogonal_matrix(&m1));
///     assert!(!is_orthogonal_matrix(&m2));
/// }
/// ```
pub fn is_orthogonal_matrix<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> bool
where
    N: Real,
{
    let transposed = transpose(m);
    if R < C {
        is_identity_matrix(&(m.clone() * transposed))
    } else {
        is_identity_matrix(&(transposed * m.clone()))
    }
}

/// Calculate the induced L-1 norm of the matrix.
///
/// The induced L-1 norm of a matrix is defined as
/// the maximum sum of the absolute values of the elements of its column vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::induced_l1_matrix_norm, matrix::Matrix},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m =
///         Matrix::<Double, 3, 2>::of(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(|e| Double::of(e)))
///             .unwrap();
///     let l1_norm = induced_l1_matrix_norm(&m);
///     assert_eq!(l1_norm, Double::of(12.0));
/// }
/// ```
pub fn induced_l1_matrix_norm<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> N
where
    N: Real,
{
    let mut norm = N::zero();
    for e in (1..=C).map(|p| {
        m.get_column(p)
            .map(|c| {
                c.linear_iter()
                    .map(|e| e.absolute_value())
                    .fold(N::zero(), |acc, e| acc + e)
            })
            .expect(concat!(
                "Error[matrix::math::induced_l1_matrix_norm]: ",
                "Failed to retrieve column vectors of the matrix."
            ))
    }) {
        if e > norm {
            norm = e;
        }
    }
    norm
}

/// Calculate the induced L-inf norm of the matrix.
///
/// The induced L-inf norm of a matrix is defined as
/// the maximum sum of the absolute values of the elements of its row vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::induced_l_inf_matrix_norm, matrix::Matrix},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m =
///         Matrix::<Double, 3, 2>::of(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(|e| Double::of(e)))
///             .unwrap();
///     let l_inf_norm = induced_l_inf_matrix_norm(&m);
///     assert_eq!(l_inf_norm, Double::of(11.0));
/// }
/// ```
pub fn induced_l_inf_matrix_norm<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> N
where
    N: Real,
{
    let mut norm = N::zero();
    for e in (1..=R).map(|p| {
        m.get_row(p)
            .map(|r| {
                r.linear_iter()
                    .map(|e| e.absolute_value())
                    .fold(N::zero(), |acc, e| acc + e)
            })
            .expect(concat!(
                "Error[matrix::math::induced_l_inf_matrix_norm]: ",
                "Failed to retrieve row vectors of the matrix."
            ))
    }) {
        if e > norm {
            norm = e;
        }
    }
    norm
}

/// Calculate the Frobenius norm of the matrix.
///
/// Both real matrix and complex matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::frobenius_norm, matrix::Matrix},
///     number::{instances::float::Float, traits::floating::Floating},
/// };
///
/// fn main() {
///     let m = Matrix::<Float, 2, 2>::of(&[1.0, 2.0, 3.0, 4.0].map(Float::of)).unwrap();
///     let f_norm = frobenius_norm(&m);
///     assert_eq!(f_norm, Float::of(30.0).square_root());
/// }
/// ```
pub fn frobenius_norm<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> N
where
    N: Floating,
{
    m.inner
        .par_iter()
        .map(|e| {
            // Important for complex numbers.
            let abs = e.absolute_value();
            abs.clone() * abs
        })
        .reduce(|| N::zero(), |a, b| a + b)
        .square_root()
}

/// Calculate the row-reduced form of the matrix.
///
/// # Returns
///
/// - Times of row swaps
/// - Permutation matrix
/// - Lower triangular matrix
/// - Row-reduced matrix
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::row_reduce, matrix::Matrix},
///     number::instances::double::Double,
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
///     assert_eq!(
///         reduced,
///         Matrix::<Double, 3, 3>::of(
///             &[
///                 2.0, 1.0, -1.0, // r1
///                 0.0, 0.5, 0.5, // r2
///                 0.0, 0.0, -1.0 // r3
///             ]
///             .map(|e| Double::of(e))
///         )
///         .unwrap()
///     );
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

/// Calculate the row-eliminate form of the matrix (inner).
///
/// This transformation matrix is the inverse of the matrix itself,
/// provided that the matrix is invertible.
///
/// # Returns
///
/// - Transform matrix
/// - Row-eliminated matrix
fn row_eliminate_inner<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> (Matrix<N, R, R>, Matrix<N, R, C>)
where
    N: Fractional,
{
    let (_, p, lt, mut eliminate) = row_reduce(&m);
    // Record the elimination process.
    let mut trans = p * lt;
    for row in (1..=R).rev() {
        let mut column = 1;
        // Find the pivot of this row.
        while column <= C && eliminate[(row, column)].is_zero() {
            column = column + 1;
        }
        if column == C + 1 {
            // Skip the row that is all zeros.
            continue;
        } else {
            let pivot = eliminate[(row, column)].clone();
            // Start elimination from the bottom of the matrix.
            for prev in (1..row).rev() {
                let prev_pivot = eliminate[(prev, column)].clone();
                if !prev_pivot.is_zero() {
                    let factor = prev_pivot / pivot.clone();
                    let add = Matrix::<N, R, R>::p_add(row, prev, -factor);
                    trans = add.clone() * trans;
                    eliminate = add * eliminate;
                }
            }
            // Make the pivot to ONE.
            let mul = Matrix::<N, R, R>::p_muls(row, N::one() / pivot);
            trans = mul.clone() * trans;
            eliminate = mul * eliminate;
        }
    }
    // returns
    (trans, eliminate)
}

/// Calculate the row-eliminate form of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::row_eliminate, matrix::Matrix},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m = Matrix::<Double, 3, 3>::of(
///         &[2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0].map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let eliminated = row_eliminate(&m);
///     assert_eq!(eliminated, Matrix::<Double, 3, 3>::eyes());
/// }
/// ```
pub fn row_eliminate<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> Matrix<N, R, C>
where
    N: Fractional,
{
    row_eliminate_inner(m).1
}

/// Calculate the inverse of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::inverse, matrix::Matrix},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m = Matrix::<Double, 3, 3>::of(
///         &[2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0].map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let inv = inverse(&m).unwrap();
///     assert_eq!(inv * m, Matrix::<Double, 3, 3>::eyes());
/// }
/// ```
pub fn inverse<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> Option<Matrix<N, R, R>>
where
    N: Fractional,
{
    let det = determinant(m);
    if det.is_none_or(|e| e.is_zero()) {
        eprintln!(concat!(
            "Error[matrix::math::inverse]: ",
            "The singular matrix does not have an inverse."
        ));
        None
    } else {
        Some(row_eliminate_inner(m).0)
    }
}

/// Calculate the determinant of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::determinant, matrix::Matrix},
///     number::{instances::double::Double, traits::zero::Zero},
/// };
///
/// fn main() {
///     let m = Matrix::<Double, 4, 4>::of(
///         &[
///             2.0, 1.0, 3.0, 4.0, // r1
///             1.0, 0.0, 2.0, 3.0, // r2
///             0.0, 1.0, 1.0, 1.0, // r3
///             3.0, 4.0, 0.0, 2.0, // r4
///         ]
///         .map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let det = determinant(&m).unwrap();
///     assert!((det - Double::of(-8.0)).is_zero());
/// }
/// ```
pub fn determinant<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> Option<N>
where
    N: Fractional,
{
    if is_square_matrix(m) {
        let (t, _, _, reduced) = row_reduce(m);
        Some(
            (1..=R)
                .map(|index| reduced[(index, index)].clone())
                .fold(N::one(), |acc, e| acc * e.clone())
                * if t & 1 == 0 { N::one() } else { -N::one() },
        )
    } else {
        eprintln!("Error[matrix::math::determinant]: Only square matrices have determinants.");
        None
    }
}

/// Calculate the determinant of a matrix using the Leibniz formula.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::determinant_l, matrix::Matrix},
///     number::instances::int::Int,
/// };
///
/// fn main() {
///     let m = Matrix::<Int, 4, 4>::of(
///         &[1, 2, 3, 4, 1, 3, 4, 1, 1, 4, 1, 2, 1, 1, 2, 3].map(Int::of),
///     )
///     .unwrap();
///     assert_eq!(determinant_l(&m), Some(Int::of(16)));
/// }
/// ```
pub fn determinant_l<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> Option<N>
where
    N: Number,
{
    if is_square_matrix(m) {
        let mut det = N::zero();
        let permutations_of_columns = permutation(&(1..=C).collect::<Vec<usize>>());
        for p in permutations_of_columns {
            let mut item = N::one();
            for (row, &column) in (1..=R).zip(p.iter()) {
                item = item * m[(row, column)].clone();
            }
            let ic = inversion_count(&p);
            if (ic & 1) == 1 {
                item = -item;
            }
            det = det + item;
        }
        Some(det)
    } else {
        eprintln!(
            "Error[matrix::math::determinant_l]: Only square matrices have determinants."
        );
        None
    }
}

/// Calculate the adjugate matrix of the matrix.
///
/// adj(m) * m = det(m) * I
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
///     matrix::{
///         math::{adjugate_matrix, determinant},
///         matrix::Matrix,
///     },
///     number::instances::double::Double,
/// };
/// fn main() {
///     let m = Matrix::<Double, 3, 3>::of(
///         &[
///             -3.0, 2.0, -5.0, // r1
///             -1.0, 0.0, -2.0, // r2
///             3.0, -4.0, 1.0, // r3
///         ]
///         .map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let adj = adjugate_matrix(&m).unwrap();
///     let expect = Matrix::<Double, 3, 3>::of(
///         &[
///             -8.0, 18.0, -4.0, // r1
///             -5.0, 12.0, -1.0, // r2
///             4.0, -6.0, 2.0, // r3
///         ]
///         .map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let det = determinant(&m).unwrap();
///     assert_eq!(adj, expect);
///     // adj(m) * m = det(m) * I
///     assert_eq!(
///         adj.clone() * m.clone(),
///         Matrix::<Double, 3, 3>::eyes() * det
///     );
///     // adj(m) * m = m * adj(m)
///     assert_eq!(adj.clone() * m.clone(), m * adj)
/// }
/// ```
pub fn adjugate_matrix<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> Option<Matrix<N, R, R>>
where
    N: Fractional,
    [(); R - 1]:,
    [(); C - 1]:,
{
    if is_square_matrix(m) {
        let mut adjugate = Matrix::<N, R, R>::default();
        for row in 1..=R {
            for column in 1..=C {
                adjugate[(row, column)] =
                    determinant(&m.submatrix(column, row)).expect(&format!(
                        concat!(
                            "Error[matrix::math::adjugate_matrix]: ",
                            "Failed to retrieve the determinant ",
                            "of the submatrix({}, {}) of the matrix."
                        ),
                        row, column
                    )) * if (row + column) & 1 == 0 {
                        N::one()
                    } else {
                        -N::one()
                    }
            }
        }
        Some(adjugate)
    } else {
        eprintln!(
            "Error[matrix::math::adjugate_matrix]: Only square matrices have adjugate matrices."
        );
        None
    }
}

/// Calculate the PLU decomposition of the matrix.
///
/// p * m = l * u
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::plu_decomposition, matrix::Matrix},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m = Matrix::<Double, 3, 3>::of(
///         &[0.0, 5.0, 22.0 / 3.0, 4.0, 2.0, 1.0, 2.0, 7.0, 9.0].map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let (p, l, u) = plu_decomposition(&m);
///     assert_eq!(p * m, l * u);
/// }
/// ```
pub fn plu_decomposition<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> (Matrix<N, R, R>, Matrix<N, R, R>, Matrix<N, R, C>)
where
    N: Fractional,
{
    let (_, p, l_inv, reduced) = row_reduce(m);
    (
        p,
        inverse(&l_inv).expect(concat!(
            "Error[matrix::math::plu_decomposition]: ",
            "Failed to retrieve the inverse."
        )),
        reduced,
    )
}
