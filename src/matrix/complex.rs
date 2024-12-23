//! # matrix::complex
//!
//! Functions for handling vectors and matrices with complex elements.

use crate::{
    matrix::{
        math::{is_identity_matrix, is_square_matrix},
        matrix::Matrix,
        utils::{apply, transpose},
        vector::VectorC,
    },
    number::{
        instances::complex::Complex,
        traits::{floating::Floating, realfloat::RealFloat, zero::Zero},
    },
};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Obtains the conjugate transpose of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::conjugate_transpose, matrix::Matrix},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 2, 2>::of(
///         &[(1.0, 2.0), (3.0, -1.0), (4.0, 0.0), (5.0, 6.0)]
///             .map(|(real, imag)| Complex::of(Float::of(real), Float::of(imag))),
///     )
///     .unwrap();
///     let n = Matrix::<Complex<Float>, 2, 2>::of(
///         &[(1.0, -2.0), (4.0, 0.0), (3.0, 1.0), (5.0, -6.0)]
///             .map(|(real, imag)| Complex::of(Float::of(real), Float::of(imag))),
///     )
///     .unwrap();
///     assert!(conjugate_transpose(&m).equals(&n));
/// }
/// ```
pub fn conjugate_transpose<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> Matrix<Complex<F>, C, R>
where
    F: RealFloat,
{
    let transposed = transpose(m);
    apply(&transposed, |e| e.conjugate())
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
///     matrix::{complex::is_unitary_matrix, matrix::Matrix},
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
///     matrix::{complex::is_normal_matrix, matrix::Matrix},
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
///     matrix::{complex::is_hermitian_matrix, matrix::Matrix},
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

/// Calculate the dot product of a complex vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::dot_product, vector::VectorC},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let v1 = VectorC::<Complex<Float>, 4>::of(
///         &[(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
///             .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let v2 = VectorC::<Complex<Float>, 4>::of(
///         &[(3.0, -4.0), (6.0, -2.0), (1.0, 2.0), (4.0, 3.0)]
///             .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     assert_eq!(
///         dot_product(v1, v2),
///         Complex::of(Float::of(1.0), Float::of(-5.0))
///     )
/// }
/// ```
pub fn dot_product<F, const R: usize>(
    v1: VectorC<Complex<F>, R>,
    v2: VectorC<Complex<F>, R>,
) -> Complex<F>
where
    F: RealFloat,
{
    v1.inner
        .par_iter()
        .zip(v2.inner.par_iter())
        .map(|(e1, e2)| e1.clone().conjugate() * e2.clone())
        .reduce(|| Complex::<F>::zero(), |acc, e| acc + e)
}
/// Calculate the Euclidean norm for a complex vector.
///
/// Aka L2-norm.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::euclidean_norm, vector::VectorC},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// fn main() {
///     let v = VectorC::<Complex<Float>, 2>::of(
///         &[(1.0, 7.0), (2.0, -6.0)].map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     assert_eq!(
///         euclidean_norm(&v),
///         Complex::of(Float::of(3.0 * 10.0f32.sqrt()), Float::zero())
///     )
/// }
/// ```
pub fn euclidean_norm<F, const R: usize>(v: &VectorC<Complex<F>, R>) -> Complex<F>
where
    F: RealFloat,
{
    v.inner
        .par_iter()
        .map(|e| e.clone() * e.clone().conjugate())
        .reduce(|| Complex::<F>::zero(), |acc, e| acc + e)
        .square_root()
}
