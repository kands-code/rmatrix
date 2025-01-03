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
        math::inverse,
        matrix::Matrix,
        utils::{apply, gram_schmidt_process, transpose},
        vector::{VectorC, normalize},
    },
    number::{
        instances::complex::Complex,
        traits::{number::Number, realfloat::RealFloat},
    },
};

/// Compute the Kronecker product of two matrices.
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
///     matrix::{extra::kronecker_product, matrix::Matrix},
///     number::instances::int::Int,
/// };
///
/// fn main() {
///     let m1 = Matrix::<Int, 2, 2>::of(&[1, 2, 0, -1].map(Int::of)).unwrap();
///     let m2 = Matrix::<Int, 2, 3>::of(&[1, 2, 3, 4, 5, 6].map(Int::of)).unwrap();
///     let p = kronecker_product(&m1, &m2);
///     let p_expect = Matrix::<Int, 4, 6>::of(
///         &[
///             1, 2, 3, 2, 4, 6, 4, 5, 6, 8, 10, 12, 0, 0, 0, -1, -2, -3, 0, 0, 0, -4, -5, -6,
///         ]
///         .map(Int::of),
///     )
///     .unwrap();
///     assert_eq!(p, p_expect);
/// }
/// ```
pub fn kronecker_product<
    N,
    const R1: usize,
    const C1: usize,
    const R2: usize,
    const C2: usize,
>(
    m1: &Matrix<N, R1, C1>,
    m2: &Matrix<N, R2, C2>,
) -> Matrix<N, { R1 * R2 }, { C1 * C2 }>
where
    N: Number,
    [(); R1 * R2]:,
    [(); C1 * C2]:,
{
    let mut product = Matrix::default();
    for row1 in 1..=R1 {
        for column1 in 1..=C1 {
            for row2 in 1..=R2 {
                for column2 in 1..=C2 {
                    product[((row1 - 1) * R2 + row2, (column1 - 1) * C2 + column2)] =
                        m1[(row1, column1)].clone() * m2[(row2, column2)].clone();
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

/// Compute the economy-sized QR decomposition of a real matrix using Givens rotation matrices.
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
///     matrix::{extra::qr_decomposition_es, math::is_orthogonal_matrix, matrix::Matrix},
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
///     let r_expect = Matrix::<Double, 3, 3>::of(
///         &[2.0, 1.0, 2.0, 0.0, 1.0, -1.0, 0.0, 0.0, 13.0f64.sqrt()].map(Double::of),
///     )
///     .unwrap();
///     assert_eq!(r, r_expect);
/// }
/// ```
pub fn qr_decomposition_es<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> (
    Matrix<N, R, { Matrix::<Complex<N>, R, C>::get_diagonal_length() }>,
    Matrix<N, { Matrix::<Complex<N>, R, C>::get_diagonal_length() }, C>,
)
where
    N: RealFloat,
    [(); Matrix::<Complex<N>, R, C>::get_diagonal_length()]:,
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
///     let m = Matrix::<Float, 5, 2>::vandermonde(
///         &[208.0, 152.0, 113.0, 227.0, 137.0].map(Float::of),
///     )
///     .unwrap();
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
///     let m =
///         Matrix::<Float, 2, 3>::of(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(Float::of)).unwrap();
///     // b
///     let b = Matrix::<Float, 2, 1>::of(&[7.0, 8.0].map(Float::of)).unwrap();
///     let sol = linear_solve_w(&m, &b);
///     // Will return one of the possible solutions.
///     let sol_expect =
///         Matrix::<Float, 3, 1>::of(&[-3.0556, 0.1111, 3.2778].map(Float::of)).unwrap();
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
///     let m = Matrix::<Float, 2, 2>::vandermonde(&[208.0, 137.0].map(Float::of)).unwrap();
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

/// Use the QR algorithm with Givens rotation matrices
/// to compute the eigenvalues and eigenvectors of a matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         extra::eigen_system_qr,
///         matrix::Matrix,
///         utils::{apply, transpose},
///         vector::VectorC,
///     },
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let m = Matrix::<Float, 3, 3>::of(
///         &[1.0, 3.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 4.0].map(Float::of),
///     )
///     .unwrap();
///     let (es, evs) = eigen_system_qr(&(transpose(&m) * m), 1024);
///     assert_eq!(
///         apply(&es, |e| e.real.clone()),
///         VectorC::<Float, 3>::of(&[20.2907, 0.4302, 9.2791].map(Float::of)).unwrap()
///     );
///     let evs_expect = Matrix::<Float, 3, 3>::of(
///         &[
///             0.3372, -0.9225, -0.1877, 0.3819, 0.3163, -0.8684, 0.8605, 0.2211, 0.45898,
///         ]
///         .map(Float::of),
///     )
///     .unwrap();
///     assert_eq!(apply(&evs, |e| e.real.clone()), evs_expect);
/// }
/// ```
pub fn eigen_system_qr<N, const E: usize>(
    m: &Matrix<N, E, E>,
    max_iter: usize,
) -> (VectorC<Complex<N>, E>, Matrix<Complex<N>, E, E>)
where
    N: RealFloat,
{
    let complexed = apply(m, |e| Complex::of(e, N::zero()));
    complex::eigen_system_qr(&complexed, max_iter)
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
///         Matrix::<Float, 3, 2>::of(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(Float::of)).unwrap();
///     let n = induced_l2_matrix_norm(&m);
///     assert_eq!(n, Float::of(9.5255));
/// }
/// ```
pub fn induced_l2_matrix_norm<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> N
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
///     let m = Matrix::<Float, 2, 2>::of(&[2.0, 8.0, 6.0, 0.0].map(Float::of)).unwrap();
///     let (u, s, v) = singular_value_decomposition(&m);
///     let s_expect = Matrix::<Float, 2, 2>::of(
///         &[6.0 * 2.0f32.sqrt(), 0.0, 0.0, 4.0 * 2.0f32.sqrt()].map(Float::of),
///     )
///     .unwrap();
///     assert_eq!(s, s_expect);
///     assert_eq!(m, u * s * transpose(&v));
/// }
/// ```
pub fn singular_value_decomposition<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> (Matrix<N, R, R>, Matrix<N, R, C>, Matrix<N, C, C>)
where
    N: RealFloat,
{
    let left_sym = m.clone() * transpose(m);
    let right_sym = transpose(m) * m.clone();
    let edge = R.min(C);
    // Compute the eigenvectors and eigenvalues of the left matrix.
    let (sig1, u) = eigen_system_qr(&left_sym, DEFAULT_MAX_ITER);
    // Convert them to real numbers.
    let sig1 = apply(&sig1, |e| e.real.clone());
    let u = apply(&u, |e| e.real.clone());
    // Compute the eigenvectors and eigenvalues of the right matrix.
    let (sig2, v) = eigen_system_qr(&right_sym, DEFAULT_MAX_ITER);
    // Convert them to real numbers.
    let sig2 = apply(&sig2, |e| e.real.clone());
    let v = apply(&v, |e| e.real.clone());
    let mut sigma = Matrix::<N, R, C>::default();
    for idx in 1..=(R.min(C)) {
        // Take the average to reduce the error.
        sigma[(idx, idx)] = if sig1[(idx, 1)].is_zero() || sig2[(idx, 1)].is_zero() {
            N::zero()
        } else {
            ((sig1[(idx, 1)].clone() + sig2[(idx, 1)].clone()) * N::half()).square_root()
        };
    }
    // Obtain the corresponding orthogonal basis.
    let mut u = gram_schmidt_process(&u);
    let mut v = gram_schmidt_process(&v);
    if R > C {
        // Use U as a reference to correct V.
        // A^T U = V (S^T) = V S'
        // => A^T ui = si vi
        let mut modified_v = Matrix::default();
        for idx in 1..=C {
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
            for row in 1..=C {
                modified_v[(row, idx)] = vi[(row, 1)].clone();
            }
        }
        v = modified_v;
    } else {
        // Use V as a reference to correct U.
        // A V = S U
        // => A vi = si ui
        let mut modified_u = Matrix::default();
        for idx in 1..=R {
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
            for row in 1..=R {
                modified_u[(row, idx)] = ui[(row, 1)].clone();
            }
        }
        u = modified_u;
    }
    // Sort singular values.
    for idx in 1..=(edge - 1) {
        let mut max = idx;
        for p in (idx + 1)..=edge {
            if sigma[(max, max)] < sigma[(p, p)] {
                max = p;
            }
        }
        if max != idx {
            let temp = sigma[(idx, idx)].clone();
            sigma[(idx, idx)] = sigma[(max, max)].clone();
            sigma[(max, max)] = temp;
            let p_left = Matrix::<N, R, R>::p_change(idx, max);
            u = u * p_left;
            let p_right = Matrix::<N, C, C>::p_change(idx, max);
            v = v * p_right;
        }
    }
    (u, sigma, v)
}

/// Compute the Moore-Penrose inverse of a real matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         extra::moore_penrose_inverse,
///         math::{inverse, is_symmetric_matrix},
///         matrix::Matrix,
///     },
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let m = Matrix::<Float, 3, 3>::of(
///         &[2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0].map(Float::of),
///     )
///     .unwrap();
///     let mp = moore_penrose_inverse(&m);
///     let m_inv = inverse(&m).unwrap();
///     // For invertible matrices,
///     // the Moore-Penrose inverse is
///     // the corresponding matrix inverse.
///     assert_eq!(mp, m_inv);
///     // Properties that the Moore-Penrose inverse must satisfy.
///     let rect =
///         Matrix::<Float, 2, 3>::of(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(Float::of)).unwrap();
///     let rectp = moore_penrose_inverse(&rect);
///     // A . Ap . A = A
///     assert_eq!(rect.clone() * rectp.clone() * rect.clone(), rect);
///     // Ap . A . Ap = Ap
///     assert_eq!(rectp.clone() * rect.clone() * rectp.clone(), rectp);
///     // A . Ap is a symmetric matrix.
///     assert!(is_symmetric_matrix(&(rect.clone() * rectp.clone())));
///     // Ap . A is also a symmetric matrix.
///     assert!(is_symmetric_matrix(&(rectp * rect)));
/// }
/// ```
pub fn moore_penrose_inverse<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> Matrix<N, C, R>
where
    N: RealFloat,
{
    let (u, s, v) = singular_value_decomposition(m);
    let mut sp = Matrix::<N, C, R>::default();
    for p in 1..=R.min(C) {
        if s[(p, p)].is_zero() {
            sp[(p, p)] = N::zero()
        } else {
            sp[(p, p)] = s[(p, p)].clone().reciprocal();
        }
    }
    v * sp * transpose(&u)
}
