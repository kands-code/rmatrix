//! # matrix::extra
//!
//! Additional mathematical functions,
//! such as matrix decomposition, eigenvalue computation,
//! and solving systems of linear equations, etc.

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    matrix::{
        math::{inverse, row_reduce},
        matrix::Matrix,
        utils::{apply, nullspace, transpose},
        vector::{dot_product, euclidean_norm, index_c, layer_product, maximum_norm, VectorC},
    },
    number::traits::{fractional::Fractional, realfloat::RealFloat},
};

/// Calculate the PLU decomposition of the matrix.
///
/// p * m = l * u
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::plu_decomposition, matrix::Matrix},
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
            "Error[matrix::extra::plu_decomposition]: ",
            "Failed to retrieve the inverse."
        )),
        reduced,
    )
}

/// Use the Gram-Schmidt process to compute the QR decomposition of a REAL matrix.
///
/// # Examples
///
/// ## Tall matrix
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::qr_decomposition_gs, matrix::Matrix, utils::transpose},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m = Matrix::<Double, 3, 2>::of(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0].map(Double::of)).unwrap();
///     let (q, r) = qr_decomposition_gs(&m).unwrap();
///     let q_expect = Matrix::<Double, 3, 2>::of(
///         &[
///             1.0 / 2.0f64.sqrt(),
///             -1.0 / 6.0f64.sqrt(),
///             0.0,
///             (2.0 / 3.0f64).sqrt(),
///             1.0 / 2.0f64.sqrt(),
///             1.0 / 6.0f64.sqrt(),
///         ]
///         .map(Double::of),
///     )
///     .unwrap();
///     assert_eq!(q, q_expect);
///     let r_expect = Matrix::<Double, 2, 2>::of(
///         &[
///             2.0f64.sqrt(),
///             1.0 / 2.0f64.sqrt(),
///             0.0,
///             (3.0 / 2.0f64).sqrt(),
///         ]
///         .map(Double::of),
///     )
///     .unwrap();
///     assert_eq!(r, r_expect);
///     // Q^T Q = I
///     assert_eq!(transpose(&q) * q, Matrix::<Double, 2, 2>::eyes());
/// }
/// ```
///
/// ## Square matrix
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::qr_decomposition_gs, matrix::Matrix, utils::transpose},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m =
///         Matrix::<Double, 3, 3>::of(&[1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0].map(Double::of))
///             .unwrap();
///     let (q, r) = qr_decomposition_gs(&m).unwrap();
///     let q_expect = Matrix::<Double, 3, 3>::of(
///         &[
///             1.0 / 2.0f64.sqrt(),
///             1.0 / 6.0f64.sqrt(),
///             -1.0 / 3.0f64.sqrt(),
///             1.0 / 2.0f64.sqrt(),
///             -1.0 / 6.0f64.sqrt(),
///             1.0 / 3.0f64.sqrt(),
///             0.0,
///             2.0 / 6.0f64.sqrt(),
///             1.0 / 3.0f64.sqrt(),
///         ]
///         .map(Double::of),
///     )
///     .unwrap();
///     assert_eq!(q, q_expect);
///     let r_expect = Matrix::<Double, 3, 3>::of(
///         &[
///             2.0 / 2.0f64.sqrt(),
///             1.0 / 2.0f64.sqrt(),
///             1.0 / 2.0f64.sqrt(),
///             0.0,
///             3.0 / 6.0f64.sqrt(),
///             1.0 / 6.0f64.sqrt(),
///             0.0,
///             0.0,
///             2.0 / 3.0f64.sqrt(),
///         ]
///         .map(Double::of),
///     )
///     .unwrap();
///     assert_eq!(r, r_expect);
/// }
/// ```
///
/// ## Warnings
///
/// <div class="warning">
///
/// The Gram-Schmidt process,
/// when applied to non-square matrices,
/// results in `Q` that only satisfies the condition of having orthonormal columns.
/// For square matrices with linearly independent columns, i.e., full rank matrices,
/// the resulting `Q` is an orthogonal matrix.
///
/// The matrix should be a tall matrix or a square matrix, i.e., `R >= C`.
///
/// **_The Gram-Schmidt process is inherently numerically unstable._**
///
/// </div>
pub fn qr_decomposition_gs<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> Option<(Matrix<N, R, C>, Matrix<N, C, C>)>
where
    N: RealFloat,
{
    if R < C {
        eprintln!(concat!(
            "Error[matrix::extra::qr_decomposition_gs]: ",
            "The matrix should be a tall matrix or a square matrix"
        ));
        None
    } else {
        let mut q = Matrix::<N, R, C>::default();
        for colum in 1..=C {
            // A = [ a1 | a2 | ... | an ]
            let a = apply(
                &m.get_column(colum).expect(&format!(
                    concat!(
                        "Error[matrix::extra::qr_decomposition_gs]: ",
                        "Failed to retrieve the {}-th column of the matrix"
                    ),
                    colum,
                )),
                |e: &N| e.clone(),
            );
            // u1 = a1
            // uk = ak - sum((ak . en) en, {n, 1, k - 1})
            let mut u = a.clone();
            for k in 1..colum {
                // Q = [ e1 | e2 | ... | en ]
                let en = apply(
                    &q.get_column(k).expect(&format!(
                        concat!(
                            "Error[matrix::extra::qr_decomposition_gs]: ",
                            "Failed to retrieve the {}-th column of the matrix"
                        ),
                        k,
                    )),
                    |e: &N| e.clone(),
                );
                // uk(n) = uk(n - 1) - en (uk(n - 1) . en)
                u = u.clone() - en.clone() * dot_product(u, en)
            }
            let u_norm = euclidean_norm(&u);
            for row in 1..=R {
                // ek = uk / ||uk||
                q[(row, colum)] = index_c(&u, row).clone() / u_norm.clone();
            }
        }
        // R = Upper {  a_c . e_r } and R = Q^T A
        let r = transpose(&q) * m.clone();
        Some((q, r))
    }
}

/// Use the Householder method to compute the QR decomposition of a REAL matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::qr_decomposition_h, matrix::Matrix},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m = Matrix::<Double, 4, 3>::of(
///         &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 4.0].map(Double::of),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_h(&m);
///     let q_expect = Matrix::<Double, 4, 4>::of(
///         &[
///             -0.5,
///             -0.5,
///             1.0 / (2.0 * 13.0f64.sqrt()),
///             -5.0 / (2.0 * 13.0f64.sqrt()),
///             -0.5,
///             -0.5,
///             -1.0 / (2.0 * 13.0f64.sqrt()),
///             5.0 / (2.0 * 13.0f64.sqrt()),
///             -0.5,
///             0.5,
///             -5.0 / (2.0 * 13.0f64.sqrt()),
///             -1.0 / (2.0 * 13.0f64.sqrt()),
///             -0.5,
///             0.5,
///             5.0 / (2.0 * 13.0f64.sqrt()),
///             1.0 / (2.0 * 13.0f64.sqrt()),
///         ]
///         .map(Double::of),
///     )
///     .unwrap();
///     assert_eq!(q, q_expect);
///     let r_expect = Matrix::<Double, 4, 3>::of(
///         &[
///             -2.0,
///             -1.0,
///             -2.0,
///             0.0,
///             -1.0,
///             1.0,
///             0.0,
///             0.0,
///             13.0f64.sqrt(),
///             0.0,
///             0.0,
///             0.0,
///         ]
///         .map(Double::of),
///     )
///     .unwrap();
///     assert_eq!(r, r_expect);
/// }
/// ```
pub fn qr_decomposition_h<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> (Matrix<N, R, R>, Matrix<N, R, C>)
where
    N: RealFloat,
{
    let mut q = Matrix::<N, R, R>::eyes();
    let mut r = m.clone();
    let two = N::one() + N::one();
    for column in 1..=C.min(R) {
        // x = col(m, c)
        let mut x = apply(
            &r.get_column(column).expect(&format!(
                concat!(
                    "Error[matrix::extra::qr_decomposition_gs]: ",
                    "Failed to retrieve the {}-th column of the matrix"
                ),
                column,
            )),
            |e| e.clone(),
        );
        // forall ei in x where i < c is zero
        for index in 1..column {
            x[(index, 1)] = N::zero();
        }
        let x_norm = euclidean_norm(&x);
        // v = x but v[col] = x[col] + x_norm * signum(x[col])
        let mut v = x.clone();
        v[(column, 1)] = x[(column, 1)].clone() + x_norm * x[(column, 1)].sign_number();
        // p = I - 2 / (v^T . v) (v * v^T)
        let p = Matrix::<N, R, R>::eyes()
            - layer_product(v.clone(), transpose(&v)) * two.clone() / dot_product(v.clone(), v);
        // r = pn p_{n - 1} ... p1 m
        r = p.clone() * r;
        // q = p1 p2 ... pn
        q = q * p;
    }
    (q, r)
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

/// Use the Givens rotation process to compute the QR decomposition of a REAL matrix .
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::qr_decomposition_gr, matrix::Matrix},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let a = Matrix::<Double, 4, 3>::of(
///         &[
///             1.0, -1.0, 4.0, 1.0, 4.0, -2.0, 1.0, 4.0, 2.0, 1.0, -1.0, 0.0,
///         ]
///         .map(Double::of),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_gr(&a);
///     let q_expect = Matrix::<Double, 4, 4>::of(
///         &[
///             0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
///         ]
///         .map(Double::of),
///     )
///     .unwrap();
///     assert_eq!(q, q_expect);
///     let r_expect = Matrix::<Double, 4, 3>::of(
///         &[2.0, 3.0, 2.0, 0.0, 5.0, -2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0].map(Double::of),
///     )
///     .unwrap();
///     assert_eq!(r, r_expect);
/// }
/// ```
pub fn qr_decomposition_gr<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> (Matrix<N, R, R>, Matrix<N, R, C>)
where
    N: RealFloat,
{
    let mut q = Matrix::<N, R, R>::eyes();
    let mut r = m.clone();
    for column in 1..=C {
        for row in (column..R).rev() {
            if r[(row, column)].is_zero() {
                continue;
            } else {
                let rotation = r.givens_rotation(row, column);
                r = rotation.clone() * r;
                q = rotation * q;
            }
        }
    }
    (transpose(&q), r)
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
/// can be computed in conjunction with the [nullspace].
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

/// Calculate the dominant eigenvalue and the corresponding normalized eigenvector using the power method.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::eigen_system_power, matrix::Matrix, vector::VectorC},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let m =
///         Matrix::<Float, 3, 3>::of(&[1.0, 2.0, 0.0, -2.0, 1.0, 2.0, 1.0, 3.0, 1.0].map(Float::of))
///             .unwrap();
///     let (e, v) = eigen_system_power(&m);
///     assert_eq!(e, Float::of(3.0));
///     assert_eq!(
///         v,
///         VectorC::<Float, 3>::of(&[1.0, 1.0, 2.0].map(|e| Float::of(e / 6.0f32.sqrt()))).unwrap()
///     );
/// }
/// ```
pub fn eigen_system_power<N, const E: usize>(m: &Matrix<N, E, E>) -> (N, VectorC<N, E>)
where
    N: RealFloat,
{
    let mut x =
        VectorC::<N, E>::of(&(1..=E).map(|_| N::one()).collect::<Vec<N>>()).expect(concat!(
            "Error[matrix::extra::eigen_system_power]: ",
            "Failed to construct the initial vector."
        ));
    let mut xk = m.clone() * x.clone();
    let mut x_max_norm = maximum_norm(&xk);
    xk = apply(&xk, |e| e / x_max_norm.clone());
    // Start the iteration process.
    loop {
        x = xk;
        // x(k) = m x(k - 1)
        xk = m.clone() * x.clone();
        x_max_norm = maximum_norm(&xk);
        // v(k) = x(k) / abs_max(x(k))
        xk = apply(&xk, |e| e / x_max_norm.clone());
        // Exclude interference from constant multiples.
        let p = x[(1, 1)].clone() / xk[(1, 1)].clone();
        if x == xk.clone() * p {
            // Achieve the desired precision and exit.
            break;
        }
    }
    // e = ((m x) . x) / ||x||
    let eigen = dot_product(m.clone() * x.clone(), x.clone()) / dot_product(x.clone(), x);
    // Normalize the eigenvector.
    let norm = euclidean_norm(&xk);
    xk = apply(&xk, |e| e / norm.clone());
    (eigen, xk)
}

/// Calculate the eigenvalues and eigenvectors of the matrix using the QR algorithm.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{extra::eigen_system_qr, matrix::Matrix},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let m =
///         Matrix::<Float, 3, 3>::of(&[2.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 3.0].map(Float::of))
///             .unwrap();
///     let (e, ev) = eigen_system_qr(&m);
///     assert_eq!(
///         e,
///         [2.0, 3.0, 3.0]
///             .iter()
///             .map(|&e| Float::of(e))
///             .collect::<Vec<_>>()
///     );
///     assert_eq!(
///         ev,
///         Matrix::<Float, 3, 3>::of(&[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0].map(Float::of))
///             .unwrap()
///     );
/// }
/// ```
pub fn eigen_system_qr<N, const E: usize>(m: &Matrix<N, E, E>) -> (Vec<N>, Matrix<N, E, E>)
where
    N: RealFloat,
{
    let mut x = m.clone();
    let (mut q, mut r) = qr_decomposition_gr(&x);
    let mut xk = r * q;
    loop {
        (q, r) = qr_decomposition_gr(&xk);
        x = xk;
        xk = r * q;
        if x == xk {
            break;
        }
    }
    let eigenvalues = (1..=E).map(|p| xk[(p, p)].clone()).collect::<Vec<_>>();
    let mut eigenvectors = Matrix::<N, E, E>::default();
    for column in 0..E {
        let v = nullspace(&(m.clone() - Matrix::eyes() * eigenvalues[column].clone()))[0].clone();
        for row in 1..=E {
            eigenvectors[(row, column + 1)] = index_c(&v, row).clone();
        }
    }
    (eigenvalues, eigenvectors)
}
