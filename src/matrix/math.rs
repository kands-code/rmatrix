//! # matrix::math
//!
//! Some mathematical functions for matrix operations,
//! such as matrix validation, matrix simplification,
//! and determinant calculation, etc.

use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelIterator,
    IntoParallelRefIterator,
    ParallelIterator,
};

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
///     let m1 = Matrix::<Word8>::of(2, 2, &[1, 2, 2, 1].map(Word8::of)).unwrap();
///     let m2 = Matrix::<Word8>::of(2, 1, &[1, 3].map(Word8::of)).unwrap();
///     assert!(is_square_matrix(&m1));
///     assert!(!is_square_matrix(&m2));
/// }
/// ```
pub const fn is_square_matrix<N>(m: &Matrix<N>) -> bool { m.row == m.column }

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
///     let m1 = Matrix::<Word8>::of(2, 2, &[1, 2, 2, 1].map(Word8::of)).unwrap();
///     let m2 = Matrix::<Word8>::of(2, 2, &[1, 2, 1, 1].map(Word8::of)).unwrap();
///     let m3 = Matrix::<Double>::of(2, 2, &[1.0, 2.0, 2.0, 1.0].map(Double::of)).unwrap();
///     assert!(is_symmetric_matrix(&m1));
///     assert!(!is_symmetric_matrix(&m2));
///     assert!(is_symmetric_matrix(&m3));
/// }
/// ```
pub fn is_symmetric_matrix<N>(m: &Matrix<N>) -> bool
where
    N: PartialEq + Sync,
{
    is_square_matrix(m)
        && points_2d((1, m.row), (1, m.column), |row, col| row < col)
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
///         Matrix::<Int8>::of(2, 2, &[Int8::of(0), Int8::of(2), Int8::of(-2), Int8::of(0)])
///             .unwrap();
///     let m2 =
///         Matrix::<Int8>::of(2, 2, &[Int8::of(1), Int8::of(2), Int8::of(-2), Int8::of(1)])
///             .unwrap();
///     let m3 = Matrix::<Double>::of(2, 2, &[0.0, 0.0, 0.0, 0.0].map(Double::of)).unwrap();
///     assert!(is_anti_symmetric_matrix(&m1));
///     assert!(!is_anti_symmetric_matrix(&m2));
///     assert!(is_anti_symmetric_matrix(&m3));
/// }
/// ```
pub fn is_anti_symmetric_matrix<N>(m: &Matrix<N>) -> bool
where
    N: Real,
{
    is_square_matrix(m)
        && points_2d((1, m.row), (1, m.column), |row, col| row <= col)
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
///     let m1: Matrix<Word8> = Matrix::of(2, 2, &[1, 1, 0, 1].map(Word8::of)).unwrap();
///     assert!(is_upper_triangular_matrix(&m1));
///     let m2: Matrix<Word8> = Matrix::of(2, 2, &[1, 0, 1, 1].map(Word8::of)).unwrap();
///     assert!(!is_upper_triangular_matrix(&m2));
/// }
/// ```
pub fn is_upper_triangular_matrix<N>(m: &Matrix<N>) -> bool
where
    N: Number,
{
    points_2d((1, m.row), (1, m.column), |row, col| row > col)
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
///     let m1: Matrix<Word8> = Matrix::of(2, 2, &[1, 1, 0, 1].map(Word8::of)).unwrap();
///     assert!(!is_lower_triangular_matrix(&m1));
///     let m2: Matrix<Word8> = Matrix::of(2, 2, &[1, 0, 1, 1].map(Word8::of)).unwrap();
///     assert!(is_lower_triangular_matrix(&m2));
/// }
/// ```
pub fn is_lower_triangular_matrix<N>(m: &Matrix<N>) -> bool
where
    N: Number,
{
    points_2d((1, m.row), (1, m.column), |row, col| row < col)
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
///     let m1: Matrix<Word8> = Matrix::of(2, 2, &[1, 0, 0, 2].map(Word8::of)).unwrap();
///     assert!(is_diagonal_matrix(&m1));
///
///     let m2: Matrix<Word8> = Matrix::of(2, 2, &[1, 0, 1, 1].map(Word8::of)).unwrap();
///     assert!(!is_diagonal_matrix(&m2));
/// }
/// ```
pub fn is_diagonal_matrix<N>(m: &Matrix<N>) -> bool
where
    N: Number,
{
    points_2d((1, m.row), (1, m.column), |row, col| row != col)
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
///     let m1 = Matrix::<Word8>::of(2, 2, &[1, 0, 0, 1].map(Word8::of)).unwrap();
///     let m2 =
///         Matrix::<Double>::of(2, 2, &[1.0, 2.0, 2.0, 1.0].map(|e| Double::of(e))).unwrap();
///     assert!(is_identity_matrix(&m1));
///     assert!(!is_identity_matrix(&m2));
///     assert!(is_identity_matrix(&Matrix::<Double>::eyes(3, 3)));
/// }
/// ```
pub fn is_identity_matrix<N>(m: &Matrix<N>) -> bool
where
    N: Number,
{
    is_square_matrix(m)
        && is_diagonal_matrix(m)
        && points_2d((1, m.row), (1, m.column), |row, col| row == col)
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
///     let m = Matrix::<Float>::of(2, 2, &[1.0, 2.0, -2.0, 1.0].map(Float::of)).unwrap();
///     assert!(is_normal_matrix(&m));
/// }
/// ```
pub fn is_normal_matrix<N>(m: &Matrix<N>) -> bool
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
///     let m1 = Matrix::<Double>::eyes(3, 3);
///     let m2 =
///         Matrix::<Double>::of(2, 2, &[1.0, 2.0, 2.0, 3.0].map(|e| Double::of(e))).unwrap();
///     assert!(is_orthogonal_matrix(&m1));
///     assert!(!is_orthogonal_matrix(&m2));
/// }
/// ```
pub fn is_orthogonal_matrix<N>(m: &Matrix<N>) -> bool
where
    N: Real,
{
    let transposed = transpose(m);
    if m.row < m.column {
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
///         Matrix::<Double>::of(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(|e| Double::of(e)))
///             .unwrap();
///     let l1_norm = induced_l1_matrix_norm(&m);
///     assert_eq!(l1_norm, Double::of(12.0));
/// }
/// ```
pub fn induced_l1_matrix_norm<N>(m: &Matrix<N>) -> N
where
    N: Real,
{
    let mut norm = N::zero();
    for e in (1..=m.column).map(|p| {
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
///         Matrix::<Double>::of(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(|e| Double::of(e)))
///             .unwrap();
///     let l_inf_norm = induced_l_inf_matrix_norm(&m);
///     assert_eq!(l_inf_norm, Double::of(11.0));
/// }
/// ```
pub fn induced_l_inf_matrix_norm<N>(m: &Matrix<N>) -> N
where
    N: Real,
{
    let mut norm = N::zero();
    for e in (1..=m.row).map(|p| {
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
///     let m = Matrix::<Float>::of(2, 2, &[1.0, 2.0, 3.0, 4.0].map(Float::of)).unwrap();
///     let f_norm = frobenius_norm(&m);
///     assert_eq!(f_norm, Float::of(30.0).square_root());
/// }
/// ```
pub fn frobenius_norm<N>(m: &Matrix<N>) -> N
where
    N: Floating,
{
    m.inner
        .par_iter()
        .take(m.row * m.column)
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
///     let m = Matrix::<Double>::of(
///         3,
///         3,
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
///         Matrix::<Double>::of(
///             3,
///             3,
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
pub fn row_reduce<N>(m: &Matrix<N>) -> (usize, Matrix<N>, Matrix<N>, Matrix<N>)
where
    N: Fractional,
{
    let mut t = 0;
    let mut p = Matrix::<N>::eyes(m.row, m.row);
    let mut lt = Matrix::<N>::eyes(m.row, m.row);
    let mut reduced = m.clone();

    if !(is_upper_triangular_matrix(&reduced) || m.row < 2) {
        // Deviation of the pivot.
        let mut deviation = 0;
        for row in 1..m.row {
            // Boundary check.
            if row + deviation > m.column {
                break;
            }
            // Pivot check.
            let mut pivot_check = reduced[(row, row + deviation)].is_zero();
            while pivot_check && (row + deviation) <= m.column {
                // Find the row where the pivot at the corresponding position is non-zero.
                for next in (row + 1)..=m.row {
                    // Perform row swapping.
                    if !reduced[(next, row + deviation)].is_zero() {
                        let swap = Matrix::<N>::p_change(m.row, m.row, row, next);
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
                for next in (row + 1)..=m.row {
                    let next_pivot = reduced[(next, row + deviation)].clone();
                    if !next_pivot.is_zero() {
                        let factor = next_pivot / pivot.clone();
                        let add = Matrix::<N>::p_add(m.row, m.row, row, next, -factor);
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
fn row_eliminate_inner<N>(m: &Matrix<N>) -> (Matrix<N>, Matrix<N>)
where
    N: Fractional,
{
    let (_, p, lt, mut eliminate) = row_reduce(&m);
    // Record the elimination process.
    let mut trans = p * lt;
    for row in (1..=m.row).rev() {
        let mut column = 1;
        // Find the pivot of this row.
        while column <= m.column && eliminate[(row, column)].is_zero() {
            column = column + 1;
        }
        if column == m.column + 1 {
            // Skip the row that is all zeros.
            continue;
        } else {
            let pivot = eliminate[(row, column)].clone();
            // Start elimination from the bottom of the matrix.
            for prev in (1..row).rev() {
                let prev_pivot = eliminate[(prev, column)].clone();
                if !prev_pivot.is_zero() {
                    let factor = prev_pivot / pivot.clone();
                    let add = Matrix::<N>::p_add(m.row, m.row, row, prev, -factor);
                    trans = add.clone() * trans;
                    eliminate = add * eliminate;
                }
            }
            // Make the pivot to ONE.
            let mul = Matrix::<N>::p_muls(m.row, m.row, row, N::one() / pivot);
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
///     let m = Matrix::<Double>::of(
///         3,
///         3,
///         &[2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0].map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let eliminated = row_eliminate(&m);
///     assert_eq!(eliminated, Matrix::<Double>::eyes(3, 3));
/// }
/// ```
pub fn row_eliminate<N>(m: &Matrix<N>) -> Matrix<N>
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
///     let m = Matrix::<Double>::of(
///         3,
///         3,
///         &[2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0].map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let inv = inverse(&m).unwrap();
///     assert_eq!(inv * m, Matrix::<Double>::eyes(3, 3));
/// }
/// ```
pub fn inverse<N>(m: &Matrix<N>) -> Option<Matrix<N>>
where
    N: Fractional,
{
    if is_square_matrix(m) {
        let det = determinant(m);
        if det.is_zero() {
            eprintln!(concat!(
                "Error[matrix::math::inverse]: ",
                "The singular matrix does not have an inverse."
            ));
            None
        } else {
            Some(row_eliminate_inner(m).0)
        }
    } else {
        eprintln!(concat!(
            "Error[matrix::math::inverse]: ",
            "Non-square matrices are not invertible."
        ));
        None
    }
}

/// Calculate the determinant of the matrix.
///
/// # Panics
///
/// Only square matrices can have a determinant calculated.
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
///     let m = Matrix::<Double>::of(
///         4,
///         4,
///         &[
///             2.0, 1.0, 3.0, 4.0, // r1
///             1.0, 0.0, 2.0, 3.0, // r2
///             0.0, 1.0, 1.0, 1.0, // r3
///             3.0, 4.0, 0.0, 2.0, // r4
///         ]
///         .map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let det = determinant(&m);
///     assert!((det - Double::of(-8.0)).is_zero());
/// }
/// ```
pub fn determinant<N>(m: &Matrix<N>) -> N
where
    N: Fractional,
{
    if is_square_matrix(m) {
        let (t, _, _, reduced) = row_reduce(m);
        (1..=m.row)
            .map(|index| reduced[(index, index)].clone())
            .fold(N::one(), |acc, e| acc * e.clone())
            * if t & 1 == 0 { N::one() } else { -N::one() }
    } else {
        panic!(concat!(
            "Error[matrix::math::determinant]: ",
            "Only square matrices can have a determinant calculated."
        ));
    }
}

/// Calculate the determinant of a matrix using the Leibniz formula.
///
/// # Panics
///
/// Only square matrices can have a determinant calculated.
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
///     let m = Matrix::<Int>::of(
///         4,
///         4,
///         &[1, 2, 3, 4, 1, 3, 4, 1, 1, 4, 1, 2, 1, 1, 2, 3].map(Int::of),
///     )
///     .unwrap();
///     assert_eq!(determinant_l(&m), Int::of(16));
/// }
/// ```
pub fn determinant_l<N>(m: &Matrix<N>) -> N
where
    N: Number,
{
    if is_square_matrix(m) {
        let mut det = N::zero();
        let permutations_of_columns = permutation(&(1..=m.column).collect::<Vec<usize>>());
        for p in permutations_of_columns {
            let mut item = N::one();
            for (row, &column) in (1..=m.row).zip(p.iter()) {
                item = item * m[(row, column)].clone();
            }
            let ic = inversion_count(&p);
            if (ic & 1) == 1 {
                item = -item;
            }
            det = det + item;
        }
        det
    } else {
        panic!(concat!(
            "Error[matrix::math::determinant_l]: ",
            "Only square matrices can have a determinant calculated."
        ));
    }
}

/// Calculate the adjugate matrix of the matrix.
///
/// adj(m) * m = det(m) * I
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         math::{adjugate_matrix, determinant},
///         matrix::Matrix,
///     },
///     number::instances::double::Double,
/// };
/// fn main() {
///     let m = Matrix::<Double>::of(
///         3,
///         3,
///         &[
///             -3.0, 2.0, -5.0, // r1
///             -1.0, 0.0, -2.0, // r2
///             3.0, -4.0, 1.0, // r3
///         ]
///         .map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let adj = adjugate_matrix(&m).unwrap();
///     let expect = Matrix::<Double>::of(
///         3,
///         3,
///         &[
///             -8.0, 18.0, -4.0, // r1
///             -5.0, 12.0, -1.0, // r2
///             4.0, -6.0, 2.0, // r3
///         ]
///         .map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let det = determinant(&m);
///     assert_eq!(adj, expect);
///     // adj(m) * m = det(m) * I
///     assert_eq!(adj.clone() * m.clone(), Matrix::<Double>::eyes(3, 3) * det);
///     // adj(m) * m = m * adj(m)
///     assert_eq!(adj.clone() * m.clone(), m * adj)
/// }
/// ```
pub fn adjugate_matrix<N>(m: &Matrix<N>) -> Option<Matrix<N>>
where
    N: Fractional,
{
    if is_square_matrix(m) {
        let mut adjugate = Matrix::<N>::defaults(m.column, m.row);
        for row in 1..=m.column {
            for column in 1..=m.row {
                adjugate[(row, column)] = determinant(&m.submatrix(column, row))
                    * if (row + column) & 1 == 0 {
                        N::one()
                    } else {
                        -N::one()
                    }
            }
        }
        Some(adjugate)
    } else {
        eprintln!(concat!(
            "Error[matrix::math::adjugate_matrix]: ",
            "Only square matrices have adjugate matrices."
        ));
        None
    }
}

/// Calculating the determinant of a matrix using the Laplace expansion method.
///
/// # Panics
///
/// Only square matrices can have a determinant calculated.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{math::determinant_e, matrix::Matrix},
///     number::instances::int8::Int8,
/// };
///
/// fn main() {
///     let m = Matrix::of(
///         4,
///         4,
///         &[1, 2, 3, 4, 1, 3, 4, 1, 1, 4, 1, 2, 1, 1, 2, 3].map(Int8::of),
///     )
///     .unwrap();
///     assert_eq!(determinant_e(&m), Int8::of(16));
/// }
/// ```
pub fn determinant_e<N>(m: &Matrix<N>) -> N
where
    N: Number,
{
    if m.row == m.column {
        match m.row {
            1 => m[(1, 1)].clone(),
            2 => m[(1, 1)].clone() * m[(2, 2)].clone() - m[(1, 2)].clone() * m[(2, 1)].clone(),
            3 => {
                m[(1, 1)].clone()
                    * (m[(2, 2)].clone() * m[(3, 3)].clone()
                        - m[(2, 3)].clone() * m[(3, 2)].clone())
                    - m[(1, 2)].clone()
                        * (m[(2, 1)].clone() * m[(3, 3)].clone()
                            - m[(2, 3)].clone() * m[(3, 1)].clone())
                    + m[(1, 3)].clone()
                        * (m[(2, 1)].clone() * m[(3, 2)].clone()
                            - m[(2, 2)].clone() * m[(3, 1)].clone())
            }
            _ => (1..=m.column)
                .into_par_iter()
                .map(|column| {
                    let submat = m.submatrix(1, column);
                    determinant_e(&submat)
                        * if ((1 + column) & 1) == 0 {
                            m[(1, column)].clone()
                        } else {
                            -m[(1, column)].clone()
                        }
                })
                .reduce(|| N::zero(), |a, b| a + b),
        }
    } else {
        panic!(concat!(
            "Error[matrix::math::determinant_e]: ",
            "Only square matrices can have a determinant calculated."
        ));
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
///     let m = Matrix::<Double>::of(
///         3,
///         3,
///         &[0.0, 5.0, 22.0 / 3.0, 4.0, 2.0, 1.0, 2.0, 7.0, 9.0].map(|e| Double::of(e)),
///     )
///     .unwrap();
///     let (p, l, u) = plu_decomposition(&m);
///     assert_eq!(p * m, l * u);
/// }
/// ```
pub fn plu_decomposition<N>(m: &Matrix<N>) -> (Matrix<N>, Matrix<N>, Matrix<N>)
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
