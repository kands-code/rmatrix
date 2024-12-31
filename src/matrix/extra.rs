//! # matrix::extra
//!
//! Additional mathematical functions,
//! such as matrix decomposition, eigenvalue computation,
//! and solving systems of linear equations, etc.

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    matrix::{
        DEFAULT_MAX_ITER,
        complex,
        math::{inverse, is_square_matrix},
        matrix::Matrix,
        utils::{apply, gram_schmidt_process, transpose},
        vector::normalize,
    },
    number::{
        instances::complex::Complex,
        traits::{number::Number, realfloat::RealFloat},
    },
};

/// Compute the Kronecker product of two matrices.
///
/// # Examples
///
/// ```rust
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use rmatrix_ks::{
///     matrix::{extra::kronecker_product, matrix::Matrix},
///     number::instances::int::Int,
/// };
///
/// fn main() {
///     let m1 = Matrix::<Int>::of(2, 2, &[1, 2, 0, -1].map(Int::of)).unwrap();
///     let m2 = Matrix::<Int>::of(2, 3, &[1, 2, 3, 4, 5, 6].map(Int::of)).unwrap();
///     let p = kronecker_product(&m1, &m2);
///     let p_expect = Matrix::<Int>::of(
///         4,
///         6,
///         &[
///             1, 2, 3, 2, 4, 6, 4, 5, 6, 8, 10, 12, 0, 0, 0, -1, -2, -3, 0, 0, 0, -4, -5, -6,
///         ]
///         .map(Int::of),
///     )
///     .unwrap();
///     assert_eq!(p, p_expect);
/// }
/// ```
pub fn kronecker_product<N>(m1: &Matrix<N>, m2: &Matrix<N>) -> Matrix<N>
where
    N: Number,
{
    let mut product = Matrix::defaults(m1.row * m2.row, m1.column * m2.column);
    for row1 in 1..=m1.row {
        for column1 in 1..=m1.column {
            for row2 in 1..=m2.row {
                for column2 in 1..=m2.column {
                    product[(
                        (row1 - 1) * m2.row + row2,
                        (column1 - 1) * m2.column + column2,
                    )] = m1[(row1, column1)].clone() * m2[(row2, column2)].clone();
                }
            }
        }
    }
    product
}

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
///     let m = Matrix::<Float>::of(
///         3,
///         3,
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
pub fn qr_decomposition_gs<N>(m: &Matrix<N>) -> Option<(Matrix<N>, Matrix<N>)>
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
///     let m = Matrix::<Float>::of(
///         3,
///         3,
///         &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0].map(Float::of),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_h(&m);
///     assert!(is_orthogonal_matrix(&q));
///     assert!(is_upper_triangular_matrix(&r));
///     assert_eq!(q * r, m);
/// }
/// ```
pub fn qr_decomposition_h<N>(m: &Matrix<N>) -> (Matrix<N>, Matrix<N>)
where
    N: RealFloat,
{
    let complexed = apply(m, |e| Complex::of(e, N::zero()));
    let (q, r) = complex::qr_decomposition_h(&complexed);
    (apply(&q, |e| e.real), apply(&r, |e| e.real))
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
///     let m = Matrix::<Float>::of(
///         3,
///         3,
///         &[12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0].map(Float::of),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_gr(&m);
///     assert!(is_orthogonal_matrix(&q));
///     assert!(is_upper_triangular_matrix(&r));
///     assert_eq!(q * r, m);
/// }
/// ```
pub fn qr_decomposition_gr<N>(m: &Matrix<N>) -> (Matrix<N>, Matrix<N>)
where
    N: RealFloat,
{
    let complexed = apply(m, |e| Complex::of(e, N::zero()));
    let (q, r) = complex::qr_decomposition_gr(&complexed);
    (apply(&q, |e| e.real), apply(&r, |e| e.real))
}

/// Compute the economy-sized QR decomposition of a real matrix using Givens rotation matrices.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::qr_decomposition_es, math::is_orthogonal_matrix, matrix::Matrix},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m = Matrix::<Double>::of(
///         4,
///         3,
///         &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 4.0].map(Double::of),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_es(&m);
///     let q_expect = Matrix::<Double>::of(
///         4,
///         3,
///         &[
///             0.5,
///             0.5,
///             1.0 / (2.0 * 13.0f64.sqrt()),
///             0.5,
///             0.5,
///             -1.0 / (2.0 * 13.0f64.sqrt()),
///             0.5,
///             -0.5,
///             -5.0 / (2.0 * 13.0f64.sqrt()),
///             0.5,
///             -0.5,
///             5.0 / (2.0 * 13.0f64.sqrt()),
///         ]
///         .map(Double::of),
///     )
///     .unwrap();
///     assert!(is_orthogonal_matrix(&q));
///     assert_eq!(q, q_expect);
///     let r_expect = Matrix::<Double>::of(
///         3,
///         3,
///         &[2.0, 1.0, 2.0, 0.0, 1.0, -1.0, 0.0, 0.0, 13.0f64.sqrt()].map(Double::of),
///     )
///     .unwrap();
///     assert_eq!(r, r_expect);
/// }
/// ```
pub fn qr_decomposition_es<N>(m: &Matrix<N>) -> (Matrix<N>, Matrix<N>)
where
    N: RealFloat,
{
    let complexed = apply(m, |e| Complex::of(e, N::zero()));
    let (q, r) = complex::qr_decomposition_es(&complexed);
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
///     let m = Matrix::<Float>::vandermonde(
///         10,
///         2,
///         &[
///             208.0, 152.0, 113.0, 227.0, 137.0, 238.0, 178.0, 104.0, 191.0, 130.0,
///         ]
///         .map(Float::of),
///     )
///     .unwrap();
///     // y
///     let b = Matrix::<Float>::of(
///         10,
///         1,
///         &[21.6, 15.5, 10.4, 31.0, 13.0, 32.4, 19.0, 10.4, 19.0, 11.8].map(Float::of),
///     )
///     .unwrap();
///     // y = sol[0] + sol[1] x
///     let sol = linear_solve_t(&m, &b);
///     let sol_expect = Matrix::<Float>::of(2, 1, &[-8.6451, 0.1612].map(Float::of)).unwrap();
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
///     let m = Matrix::<Float>::vandermonde(
///         5,
///         2,
///         &[208.0, 152.0, 113.0, 227.0, 137.0].map(Float::of),
///     )
///     .unwrap();
///     // b
///     let b = Matrix::<Float>::defaults(5, 1);
///     let sol = linear_solve_t(&m, &b);
///     // Should return a zero matrix.
///     let sol_expect = Matrix::<Float>::defaults(2, 1);
///     assert_eq!(sol, sol_expect);
/// }
/// ```
///
/// </div>
pub fn linear_solve_t<N>(m: &Matrix<N>, b: &Matrix<N>) -> Matrix<N>
where
    N: RealFloat,
{
    if b.inner.par_iter().all(|e| e.is_zero()) {
        Matrix::defaults(m.column, b.column)
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
///     let m =
///         Matrix::<Float>::of(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(Float::of)).unwrap();
///     // b
///     let b = Matrix::<Float>::of(2, 1, &[7.0, 8.0].map(Float::of)).unwrap();
///     let sol = linear_solve_w(&m, &b);
///     // Will return one of the possible solutions.
///     let sol_expect =
///         Matrix::<Float>::of(3, 1, &[-3.0556, 0.1111, 3.2778].map(Float::of)).unwrap();
///     assert_eq!(sol, sol_expect);
/// }
/// ```
///
/// The information for the complete solution
/// can be computed in conjunction with the [null_space](crate::matrix::utils::null_space).
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
///     let m = Matrix::<Float>::vandermonde(2, 2, &[208.0, 137.0].map(Float::of)).unwrap();
///     // b
///     let b = Matrix::<Float>::defaults(2, 1);
///     let sol = linear_solve_w(&m, &b);
///     // Should return a zero matrix.
///     let sol_expect = Matrix::<Float>::defaults(2, 1);
///     assert_eq!(sol, sol_expect);
/// }
/// ```
///
/// </div>
pub fn linear_solve_w<N>(m: &Matrix<N>, b: &Matrix<N>) -> Matrix<N>
where
    N: RealFloat,
{
    if b.inner.par_iter().all(|e| e.is_zero()) {
        Matrix::defaults(m.column, b.column)
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

/// Use the QR algorithm with Givens rotation matrices
/// to compute the eigenvalues and eigenvectors of a matrix.
///
/// # Panics
///
/// Only square matrices can potentially have eigenvalues.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         extra::eigen_system_qr,
///         matrix::Matrix,
///         utils::{apply, transpose},
///         vector::column_vector,
///     },
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let m = Matrix::<Float>::of(
///         3,
///         3,
///         &[1.0, 3.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 4.0].map(Float::of),
///     )
///     .unwrap();
///     let (es, evs) = eigen_system_qr(&(transpose(&m) * m), 1024);
///     assert_eq!(
///         apply(&es, |e| e.real.clone()),
///         column_vector::<Float>(3, &[20.2907, 0.4302, 9.2791].map(Float::of)).unwrap()
///     );
///     let evs_expect = Matrix::<Float>::of(
///         3,
///         3,
///         &[
///             0.39185, -4.17221, -0.40899, 0.44383, 1.43042, -1.89201, 1.0, 1.0, 1.0,
///         ]
///         .map(Float::of),
///     )
///     .unwrap();
///     assert_eq!(apply(&evs, |e| e.real.clone()), evs_expect);
/// }
/// ```
pub fn eigen_system_qr<N>(
    m: &Matrix<N>,
    max_iter: usize,
) -> (Matrix<Complex<N>>, Matrix<Complex<N>>)
where
    N: RealFloat,
{
    if is_square_matrix(m) {
        let complexed = apply(m, |e| Complex::of(e, N::zero()));
        complex::eigen_system_qr(&complexed, max_iter)
    } else {
        panic!(concat!(
            "Error[matrix::extra::eigen_system_qr]: ",
            "Only square matrices can potentially have eigenvalues."
        ))
    }
}

/// Calculate the induced L-2 norm of the real matrix.
///
/// aka. spectral norm.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::induced_l2_matrix_norm, matrix::Matrix},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let m =
///         Matrix::<Float>::of(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(Float::of)).unwrap();
///     let n = induced_l2_matrix_norm(&m);
///     assert_eq!(n, Float::of(9.5255));
/// }
/// ```
pub fn induced_l2_matrix_norm<N>(m: &Matrix<N>) -> N
where
    N: RealFloat,
{
    let p = transpose(m) * m.clone();
    let (rho_square, _) = eigen_system_qr(&p, DEFAULT_MAX_ITER);
    rho_square[(1, 1)].real.clone().square_root()
}

/// Compute the singular value decomposition of the real matrix
///
/// The orthogonal basis part is based on the Gram-Schmidt process.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::singular_value_decomposition, matrix::Matrix, utils::transpose},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let m = Matrix::<Float>::of(2, 2, &[2.0, 8.0, 6.0, 0.0].map(Float::of)).unwrap();
///     let (u, s, v) = singular_value_decomposition(&m);
///     let s_expect = Matrix::<Float>::of(
///         2,
///         2,
///         &[6.0 * 2.0f32.sqrt(), 0.0, 0.0, 4.0 * 2.0f32.sqrt()].map(Float::of),
///     )
///     .unwrap();
///     assert_eq!(s, s_expect);
///     assert_eq!(m, u * s * transpose(&v));
/// }
/// ```
pub fn singular_value_decomposition<N>(m: &Matrix<N>) -> (Matrix<N>, Matrix<N>, Matrix<N>)
where
    N: RealFloat,
{
    let left_sym = m.clone() * transpose(m);
    let right_sym = transpose(m) * m.clone();
    let edge = m.edge();
    // Compute the eigenvectors and eigenvalues of the left matrix.
    let (sig1, u) = eigen_system_qr(&left_sym, DEFAULT_MAX_ITER);
    // Convert them to real numbers.
    let sig1 = apply(&sig1, |e| e.real.clone());
    let mut u = apply(&u, |e| e.real.clone());
    // Compute the eigenvectors and eigenvalues of the right matrix.
    let (sig2, v) = eigen_system_qr(&right_sym, DEFAULT_MAX_ITER);
    // Convert them to real numbers.
    let sig2 = apply(&sig2, |e| e.real.clone());
    let mut v = apply(&v, |e| e.real.clone());
    let mut sigma = Matrix::<N>::defaults(m.row, m.column);
    for idx in 1..=edge {
        // Take the average to reduce the error.
        sigma[(idx, idx)] = if sig1[(idx, 1)].is_zero() || sig2[(idx, 1)].is_zero() {
            N::zero()
        } else {
            ((sig1[(idx, 1)].clone() + sig2[(idx, 1)].clone()) * N::half()).square_root()
        };
    }
    // Sort singular values.
    let mut sd = (1..=edge)
        .map(|idx| {
            // Take the average to reduce the error.
            if sig1[(idx, 1)].is_zero() || sig2[(idx, 1)].is_zero() {
                N::zero()
            } else {
                ((sig1[(idx, 1)].clone() + sig2[(idx, 1)].clone()) * N::half()).square_root()
            }
        })
        .collect::<Vec<N>>();
    for idx in 0..(edge - 1) {
        let mut max = idx;
        for p in (idx + 1)..edge {
            if sd[max] < sd[p] {
                max = p;
            }
        }
        if max != idx {
            let temp = sd[idx].clone();
            sd[idx] = sd[max].clone();
            sd[max] = temp;
            let p_left = Matrix::<N>::p_change(m.row, m.row, idx, max);
            u = u * p_left;
            let p_right = Matrix::<N>::p_change(m.column, m.column, idx, max);
            v = v * p_right;
        }
    }
    for idx in 1..=edge {
        sigma[(idx, idx)] = sd[idx - 1].clone();
    }
    // Obtain the corresponding orthogonal basis.
    let u = gram_schmidt_process(&u);
    let v = gram_schmidt_process(&v);
    if m.row > m.column {
        // Use U as a reference to correct V.
        // A^T U = V (S^T) = V S'
        // => A^T ui = si vi
        let mut modified_v = Matrix::defaults(m.column, m.column);
        for idx in 1..=m.column {
            let ui = apply(
                &u.get_column(idx).expect(&format!(
                    concat!(
                        "Error[matrix::extra::singular_value_decomposition]: ",
                        "Failed to obtain the {}-th column vector of U."
                    ),
                    idx
                )),
                |e: &N| e.clone(),
            );
            let vi = if sigma[(idx, idx)].is_zero() {
                normalize(&(transpose(m) * ui))
            } else {
                transpose(m) * ui / sigma[(idx, idx)].clone()
            };
            for row in 1..=m.column {
                modified_v[(row, idx)] = vi[(row, 1)].clone();
            }
        }
        (u, sigma, modified_v)
    } else {
        // Use V as a reference to correct U.
        // A V = S U
        // => A vi = si ui
        let mut modified_u = Matrix::defaults(m.row, m.row);
        for idx in 1..=m.row {
            let vi = apply(
                &v.get_column(idx).expect(&format!(
                    concat!(
                        "Error[matrix::extra::singular_value_decomposition]: ",
                        "Failed to obtain the {}-th column vector of V."
                    ),
                    idx
                )),
                |e: &N| e.clone(),
            );
            let ui = if sigma[(idx, idx)].is_zero() {
                normalize(&(m.clone() * vi))
            } else {
                m.clone() * vi / sigma[(idx, idx)].clone()
            };
            for row in 1..=m.row {
                modified_u[(row, idx)] = ui[(row, 1)].clone();
            }
        }
        (modified_u, sigma, v)
    }
}
