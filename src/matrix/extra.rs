//! # matrix::extra
//!
//! Additional mathematical functions,
//! such as matrix decomposition, eigenvalue computation,
//! and solving systems of linear equations, etc.

use crate::{
    matrix::{
        math::{inverse, row_reduce},
        matrix::Matrix,
        utils::{apply, transpose},
        vector::{dot_product, euclidean_norm, index_c, layer_product},
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
///     assert!((p * m).equals(&(l * u)));
/// }
/// ```
pub fn plu_decomposition<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> (Matrix<N, R, R>, Matrix<N, R, R>, Matrix<N, R, C>)
where
    N: Fractional,
{
    let (_, p, lt, reduced) = row_reduce(m);
    (
        p,
        inverse(&lt).expect(concat!(
            "Error[matrix::extra::plu_decomposition]: ",
            "Failed to retrieve the inverse of 'lt'."
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

/// solve linear equations
///
/// **TODO**
pub fn linear_solve<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, R>,
    b: &Matrix<N, R, C>,
) -> Matrix<N, R, C>
where
    N: Fractional,
{
    let m_inv = inverse(&m).unwrap();
    m_inv * b.clone()
}

/// eigen values
///
/// **TODO**
pub fn eigen_values() {
    todo!()
}

/// eigen vectors
///
/// **TODO**
pub fn eigen_system() {
    todo!()
}
