//! # matrix::extra
//!
//! Additional mathematical functions,
//! such as matrix decomposition, eigenvalue computation,
//! and solving systems of linear equations, etc.

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    matrix::{
        complex,
        math::inverse,
        matrix::Matrix,
        utils::{apply, transpose},
    },
    number::{instances::complex::Complex, traits::realfloat::RealFloat},
};

/// Compute the QR decomposition of a real matrix using the Gram-Schmidt process.
///
/// QR decomposition based on the Gram-Schmidt process for complex matrices.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         extra::qr_decomposition_gs,
///         math::{is_orthogonal_matrix, is_upper_triangular_matrix},
///         matrix::Matrix,
///     },
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let m = Matrix::<Float, 3, 3>::of(
///         &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0].map(Float::of),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_gs(&m).unwrap();
///     assert!(is_orthogonal_matrix(&q));
///     assert!(is_upper_triangular_matrix(&r));
///     assert_eq!(q * r, m);
/// }
/// ```
///
/// ## Warnings
///
/// <div class="warning">
///
/// **_The Gram-Schmidt process is inherently numerically unstable._**
///
/// If the input matrix is a wide matrix,
/// meaning the number of columns is greater than the number of rows,
/// the function will return None,
/// as the Gram-Schmidt process requires the columns of the matrix to be pairwise orthogonal,
/// which is not possible for wide matrices.
///
/// </div>
pub fn qr_decomposition_gs<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> Option<(Matrix<N, R, C>, Matrix<N, C, C>)>
where
    N: RealFloat,
{
    let complexed = apply(m, |e| Complex::of(e, N::zero()));
    let qr = complex::qr_decomposition_gs(&complexed);
    qr.map(|(q, r)| (apply(&q, |e| e.real), apply(&r, |e| e.real)))
}

/// Compute the QR decomposition of a real matrix using Householder transformations.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         extra::qr_decomposition_h,
///         math::{is_orthogonal_matrix, is_upper_triangular_matrix},
///         matrix::Matrix,
///     },
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let m = Matrix::<Float, 3, 3>::of(
///         &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0].map(Float::of),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_h(&m);
///     assert!(is_orthogonal_matrix(&q));
///     assert!(is_upper_triangular_matrix(&r));
///     assert_eq!(q * r, m);
/// }
/// ```
pub fn qr_decomposition_h<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> (Matrix<N, R, R>, Matrix<N, R, C>)
where
    N: RealFloat,
{
    let complexed = apply(m, |e| Complex::of(e, N::zero()));
    let (q, r) = complex::qr_decomposition_h(&complexed);
    (apply(&q, |e| e.real), apply(&r, |e| e.real))
}

/// Use the Householder method to compute the economy-size QR decomposition of a REAL matrix.
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
///     matrix::{extra::qr_decomposition_es, matrix::Matrix},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m = Matrix::<Double, 4, 3>::of(
///         &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 4.0].map(Double::of),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_es(&m);
///     let q_expect = Matrix::<Double, 4, 3>::of(
///         &[
///             -0.5,
///             -0.5,
///             1.0 / (2.0 * 13.0f64.sqrt()),
///             -0.5,
///             -0.5,
///             -1.0 / (2.0 * 13.0f64.sqrt()),
///             -0.5,
///             0.5,
///             -5.0 / (2.0 * 13.0f64.sqrt()),
///             -0.5,
///             0.5,
///             5.0 / (2.0 * 13.0f64.sqrt()),
///         ]
///         .map(Double::of),
///     )
///     .unwrap();
///     assert_eq!(q, q_expect);
///     let r_expect = Matrix::<Double, 3, 3>::of(
///         &[-2.0, -1.0, -2.0, 0.0, -1.0, 1.0, 0.0, 0.0, 13.0f64.sqrt()].map(Double::of),
///     )
///     .unwrap();
///     assert_eq!(r, r_expect);
/// }
/// ```
pub fn qr_decomposition_es<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> (
    Matrix<N, R, { Matrix::<N, R, C>::get_diagonal_length() }>,
    Matrix<N, { Matrix::<N, R, C>::get_diagonal_length() }, C>,
)
where
    N: RealFloat,
    [(); Matrix::<N, R, C>::get_diagonal_length()]:,
{
    let (basic_q, basic_r) = qr_decomposition_h(m);
    let thin = Matrix::<N, R, C>::get_diagonal_length();
    let mut q = Matrix::default();
    let mut r = Matrix::default();
    if R > C {
        // If m > n, then qr computes only the first n columns of Q and the first n rows of R.
        for row in 1..=R {
            for column in 1..=thin {
                q[(row, column)] = basic_q[(row, column)].clone();
            }
        }
        r.inner = basic_r.inner[..(thin * C)].to_vec();
    } else {
        // Else the economy-size decomposition is the same as the regular decomposition.
        q.inner = basic_q.inner;
        r.inner = basic_r.inner;
    }
    (q, r)
}

/// Compute the QR decomposition of a real matrix using Givens rotation matrices.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         extra::qr_decomposition_gr,
///         math::{is_orthogonal_matrix, is_upper_triangular_matrix},
///         matrix::Matrix,
///     },
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let m = Matrix::<Float, 3, 3>::of(
///         &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0].map(Float::of),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_gr(&m);
///     assert!(is_orthogonal_matrix(&q));
///     assert!(is_upper_triangular_matrix(&r));
///     assert_eq!(q * r, m);
/// }
/// ```
pub fn qr_decomposition_gr<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> (Matrix<N, R, R>, Matrix<N, R, C>)
where
    N: RealFloat,
{
    let complexed = apply(m, |e| Complex::of(e, N::zero()));
    let (q, r) = complex::qr_decomposition_gr(&complexed);
    (apply(&q, |e| e.real), apply(&r, |e| e.real))
}

/// Use QR decomposition to solve linear equation problems for tall matrices or square matrices.
///
/// For `M x = b`, we can have `M = Q R`, thus `x = inv(R) trans(Q) b`.
///
/// # Panics
///
/// Since the QR decomposition uses the Gram-Schmidt process,
/// the matrix must be a tall matrix or a square matrix,
/// otherwise, it will panic.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::linear_solve_t, matrix::Matrix},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     // x
///     let m = Matrix::<Float, 10, 2>::vandermonde(
///         &[
///             208.0, 152.0, 113.0, 227.0, 137.0, 238.0, 178.0, 104.0, 191.0, 130.0,
///         ]
///         .map(Float::of),
///     )
///     .unwrap();
///     // y
///     let b = Matrix::<Float, 10, 1>::of(
///         &[21.6, 15.5, 10.4, 31.0, 13.0, 32.4, 19.0, 10.4, 19.0, 11.8].map(Float::of),
///     )
///     .unwrap();
///     // y = sol[0] + sol[1] x
///     let sol = linear_solve_t(&m, &b);
///     let sol_expect = Matrix::<Float, 2, 1>::of(&[-8.6451, 0.1612].map(Float::of)).unwrap();
///     assert_eq!(sol, sol_expect);
/// }
/// ```
///
/// ## Warnings
///
/// <div class="warning">
///
/// For homogeneous systems of equations, the function will always return a zero matrix as the result.
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::linear_solve_t, matrix::Matrix},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     // M
///     let m = Matrix::<Float, 5, 2>::vandermonde(&[208.0, 152.0, 113.0, 227.0, 137.0].map(Float::of))
///         .unwrap();
///     // b
///     let b = Matrix::<Float, 5, 1>::default();
///     let sol = linear_solve_t(&m, &b);
///     // Should return a zero matrix.
///     let sol_expect = Matrix::<Float, 2, 1>::default();
///     assert_eq!(sol, sol_expect);
/// }
/// ```
///
/// </div>
pub fn linear_solve_t<N, const R: usize, const C1: usize, const C2: usize>(
    m: &Matrix<N, R, C1>,
    b: &Matrix<N, R, C2>,
) -> Matrix<N, C1, C2>
where
    N: RealFloat,
{
    if b.inner.par_iter().all(|e| e.is_zero()) {
        Matrix::default()
    } else {
        let (q, r) = qr_decomposition_gs(&m).expect(concat!(
            "Error[matrix::extra::linear_solve_t]: ",
            "Only high matrices or square matrices ",
            "can use this function to solve linear equations."
        ));
        let r_inv = inverse(&r).expect(concat!(
            "Error[matrix::extra::linear_solve_t]: ",
            "Should be able to compute the inverse of an upper triangular matrix."
        ));
        r_inv * transpose(&q) * b.clone()
    }
}

/// Use QR decomposition to solve linear equation problems for wide matrices.
///
/// For wide matrices, we can have `trans(M) = Q R`,
/// which means` M = trans(Q R) = trans(R) trans(Q)`,
/// or alternatively `M = L trans(Q)`.
/// Therefore, for the equation `M x = b`, we can express `x` as `x = Q inv(L) b`, since `trans(Q) Q = I`.
///
/// # Panics
///
/// Since the QR decomposition uses the Gram-Schmidt process,
/// the matrix must be a wide matrix or a square matrix,
/// otherwise, it will panic.
///
/// For tall matrices, please refer to [linear_solve_t].
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::linear_solve_w, matrix::Matrix},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     // M
///     let m = Matrix::<Float, 2, 3>::of(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(Float::of)).unwrap();
///     // b
///     let b = Matrix::<Float, 2, 1>::of(&[7.0, 8.0].map(Float::of)).unwrap();
///     let sol = linear_solve_w(&m, &b);
///     // Will return one of the possible solutions.
///     let sol_expect = Matrix::<Float, 3, 1>::of(&[-3.0556, 0.1111, 3.2778].map(Float::of)).unwrap();
///     assert_eq!(sol, sol_expect);
/// }
/// ```
///
/// The information for the complete solution
/// can be computed in conjunction with the [nullspace](crate::matrix::utils::nullspace).
///
/// ## Warnings
///
/// <div class="warning">
///
/// For homogeneous systems of equations, the function will always return a zero matrix as the result.
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::linear_solve_w, matrix::Matrix},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     // M
///     let m = Matrix::<Float, 2, 2>::vandermonde(&[208.0, 137.0].map(Float::of))
///         .unwrap();
///     // b
///     let b = Matrix::<Float, 2, 1>::default();
///     let sol = linear_solve_w(&m, &b);
///     // Should return a zero matrix.
///     let sol_expect = Matrix::<Float, 2, 1>::default();
///     assert_eq!(sol, sol_expect);
/// }
/// ```
///
/// </div>
pub fn linear_solve_w<N, const R: usize, const C1: usize, const C2: usize>(
    m: &Matrix<N, R, C1>,
    b: &Matrix<N, R, C2>,
) -> Matrix<N, C1, C2>
where
    N: RealFloat,
{
    if b.inner.par_iter().all(|e| e.is_zero()) {
        Matrix::default()
    } else {
        let (q, r) = qr_decomposition_gs(&transpose(m)).expect(concat!(
            "Error[matrix::extra::linear_solve_w]: ",
            "Only high matrices or square matrices ",
            "can use this function to solve linear equations."
        ));
        let l_inv = inverse(&transpose(&r)).expect(concat!(
            "Error[matrix::extra::linear_solve_w]: ",
            "Should be able to compute the inverse of an lower triangular matrix."
        ));
        q * l_inv * b.clone()
    }
}
