//! # matrix::extra
//!
//! Additional mathematical functions,
//! such as matrix decomposition, eigenvalue computation,
//! and solving systems of linear equations, etc.

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    matrix::{
        complex,
        math::{inverse, is_diagonal_matrix, is_symmetric_matrix},
        matrix::Matrix,
        utils::{apply, null_space, transpose},
        vector::VectorC,
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
/// can be computed in conjunction with the [null_space].
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
///         extra::eigen_system_symmetric,
///         matrix::Matrix,
///         utils::transpose,
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
///     let Some((es, evs)) = eigen_system_symmetric(&(transpose(&m) * m)) else {
///         unreachable!()
///     };
///     assert_eq!(
///         es,
///         VectorC::<Float, 3>::of(&[20.2907, 9.2791, 0.4302].map(Float::of)).unwrap()
///     );
///     let evs_expect = Matrix::<Float, 3, 3>::of(
///         &[
///             0.39185, -0.40899, -4.17221, 0.44383, -1.89201, 1.43042, 1.0, 1.0, 1.0,
///         ]
///         .map(Float::of),
///     )
///     .unwrap();
///     assert_eq!(evs, evs_expect);
/// }
/// ```
///
/// ## Warnings
///
/// <div class="warning">
///
/// This function can only be used to
/// compute the eigenvalues and eigenvectors of symmetric real matrices.
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::eigen_system_symmetric, matrix::Matrix},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let m = Matrix::<Float, 3, 3>::of(
///         &[1.0, 3.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 4.0].map(Float::of),
///     )
///     .unwrap();
///     // Will return None for non-symmetric matrices.
///     assert_eq!(eigen_system_symmetric(&m), None);
/// }
/// ```
///
/// </div>
pub fn eigen_system_symmetric<N, const E: usize>(
    m: &Matrix<N, E, E>,
) -> Option<(VectorC<N, E>, Matrix<N, E, E>)>
where
    N: RealFloat,
{
    if is_symmetric_matrix(m) {
        let (mut q, mut r) = qr_decomposition_gr(m);
        let mut diag = r * q;
        while !is_diagonal_matrix(&diag) {
            (q, r) = qr_decomposition_gr(&diag);
            diag = r * q;
        }
        let eigenvalues =
            VectorC::of(&(1..=E).map(|p| diag[(p, p)].clone()).collect::<Vec<N>>())
                .expect("msg");
        let mut eigenvectors = Matrix::default();
        for idx in 1..=E {
            let eigenequation = m.clone() - Matrix::eyes() * eigenvalues[(idx, 1)].clone();
            let nullspace_eq = null_space(&eigenequation);
            let eigenvector = if nullspace_eq.is_empty() {
                let (ev, _) = qr_decomposition_gr(&eigenequation);
                apply(
                    &ev.get_column(idx).expect(&format!(
                        concat!(
                            "Error[matrix::extra::eigen_system_symmetric]: ",
                            "Failed to obtain the {}-th column vector of ev."
                        ),
                        idx
                    )),
                    |e: &N| e.clone(),
                )
            } else {
                nullspace_eq[0].clone()
            };
            for row in 1..=E {
                eigenvectors[(row, idx)] = eigenvector[(row, 1)].clone();
            }
        }
        Some((eigenvalues, eigenvectors))
    } else {
        eprintln!(concat!(
            "Error[matrix::extra::eigen_system_symmetric]: ",
            "This function can only be used to compute ",
            "the eigenvalues and eigenvectors of symmetric real matrices."
        ));
        None
    }
}

/// Compute the singular value decomposition of the matrix
///
/// The orthogonal basis part is based on the Givens rotation matrices.
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
///     let u_expect = Matrix::<Float, 2, 2>::of(
///         &[
///             3.0 / 10.0f32.sqrt(),
///             1.0 / 10.0f32.sqrt(),
///             1.0 / 10.0f32.sqrt(),
///             -3.0 / 10.0f32.sqrt(),
///         ]
///         .map(Float::of),
///     )
///     .unwrap();
///     assert_eq!(u, u_expect);
///     let s_expect = Matrix::<Float, 2, 2>::of(
///         &[6.0 * 2.0f32.sqrt(), 0.0, 0.0, 4.0 * 2.0f32.sqrt()].map(Float::of),
///     )
///     .unwrap();
///     assert_eq!(s, s_expect);
///     let v_expect = Matrix::<Float, 2, 2>::of(
///         &[
///             1.0 / 5.0f32.sqrt(),
///             -2.0 / 5.0f32.sqrt(),
///             2.0 / 5.0f32.sqrt(),
///             1.0 / 5.0f32.sqrt(),
///         ]
///         .map(Float::of),
///     )
///     .unwrap();
///     assert_eq!(v, v_expect);
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
    let (sig1, u) = eigen_system_symmetric(&left_sym).expect(concat!(
        "Error[matrix::extra::singular_value_decomposition]: ",
        "Failed to compute the eigen system of (A A^T)."
    ));
    let (sig2, v) = eigen_system_symmetric(&right_sym).expect(concat!(
        "Error[matrix::extra::singular_value_decomposition]: ",
        "Failed to compute the eigen system of (A^T A)."
    ));
    let mut sigma = Matrix::<N, R, C>::default();
    for idx in 1..=(R.min(C)) {
        // Take the average to reduce the error.
        sigma[(idx, idx)] =
            ((sig1[(idx, 1)].clone() + sig2[(idx, 1)].clone()) * N::half()).square_root();
    }
    // Obtain the corresponding orthogonal basis.
    let (u, _) = qr_decomposition_gr(&u);
    let (v, _) = qr_decomposition_gr(&v);
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
            let vi = transpose(m) * ui / sigma[(idx, idx)].clone();
            for row in 1..=C {
                modified_v[(row, idx)] = vi[(row, 1)].clone();
            }
        }
        (u, sigma, modified_v)
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
            let ui = m.clone() * vi / sigma[(idx, idx)].clone();
            for row in 1..=R {
                modified_u[(row, idx)] = ui[(row, 1)].clone();
            }
        }
        (modified_u, sigma, v)
    }
}
