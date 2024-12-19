//! # matrix::extra
//!
//! Additional mathematical functions,
//! such as matrix decomposition, eigenvalue computation,
//! and solving systems of linear equations, etc.

use crate::{
    matrix::{
        math::{inverse, row_reduce},
        matrix::Matrix,
    },
    number::traits::fractional::Fractional,
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
        inverse(&lt).expect(
            "Error[matrix::extra::plu_decomposition]: Failed to retrieve the inverse of 'lt'.",
        ),
        reduced,
    )
}

/// eigen values
///
/// **TODO**
pub fn eigen_values() {
    todo!()
}

/// solve linear equations
///
/// **TODO**
pub fn linear_solve() {
    todo!()
}

/// eigen vectors
///
/// **TODO**
pub fn eigen_system() {
    todo!()
}
