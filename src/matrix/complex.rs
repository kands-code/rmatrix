//! # matrix::complex
//!
//! Functions for handling vectors and matrices with complex elements.

use crate::{
    matrix::{
        math::{is_identity_matrix, is_square_matrix},
        matrix::Matrix,
        utils::{apply, points_2d, transpose},
        vector::VectorC,
    },
    number::{
        instances::complex::Complex,
        traits::{floating::Floating, realfloat::RealFloat, zero::Zero},
    },
};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

#[cfg(feature = "extra")]
use crate::matrix::extra;

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
///     assert_eq!(conjugate_transpose(&m), n);
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
    if R < C {
        is_identity_matrix(&(m.clone() * conjugate_transposed))
    } else {
        is_identity_matrix(&(conjugate_transposed * m.clone()))
    }
}

/// Validate whether a matrix is a normal matrix for complex matrix.
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
    is_square_matrix(m)
        && points_2d((1, R), (1, C), |row, col| row <= col)
            .par_iter()
            .all(|&p @ (row, col)| m[p] == m[(col, row)].clone().conjugate())
}

/// Validate whether a matrix is an anti-hermitian matrix.
///
/// A matrix is a anti-hermitian matrix
/// if and only if the conjugate transpose of the matrix
/// is equal to the negative of the matrix itself.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::is_anti_hermitian_matrix, matrix::Matrix},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// fn main() {
///     let m1 = Matrix::<Complex<Float>, 2, 2>::of(&[
///         Complex::of(Float::zero(), Float::zero()),
///         Complex::of(Float::of(1.0f32), Float::of(1.0f32)),
///         Complex::of(Float::of(-1.0f32), Float::of(1.0f32)),
///         Complex::of(Float::zero(), Float::zero()),
///     ])
///     .unwrap();
///     let m2 = Matrix::<Complex<Float>, 2, 2>::of(&[
///         Complex::of(Float::of(1.0f32), Float::zero()),
///         Complex::of(Float::of(2.0f32), Float::of(1.0f32)),
///         Complex::of(Float::of(3.0f32), Float::zero()),
///         Complex::of(Float::of(4.0f32), Float::zero()),
///     ])
///     .unwrap();
///     assert!(is_anti_hermitian_matrix(&m1));
///     assert!(!is_anti_hermitian_matrix(&m2));
/// }
/// ```
pub fn is_anti_hermitian_matrix<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> bool
where
    F: RealFloat,
{
    is_square_matrix(m)
        && points_2d((1, R), (1, C), |row, col| row <= col)
            .par_iter()
            .all(|&p @ (row, col)| m[p] == -m[(col, row)].clone().conjugate())
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
        .map(|e| e.clone().conjugate() * e.clone())
        .reduce(|| Complex::<F>::zero(), |acc, e| acc + e)
        .square_root()
}

/// Calculate the maximum norm for complex vector.
///
/// Aka L_inf-norm.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::maximum_norm, vector::VectorC},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let m = VectorC::<Complex<Float>, 3>::of(
///         &[(1.0, 1.0), (-2.0, 0.0), (3.0, -4.0)]
///             .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let l_inf_norm = maximum_norm(&m);
///     assert_eq!(l_inf_norm, Float::of(5.0));
/// }
/// ```
pub fn maximum_norm<F, const R: usize>(v: &VectorC<Complex<F>, R>) -> F
where
    F: RealFloat,
{
    let mut norm = F::zero();
    for e in v.linear_iter().map(|e| e.clone().norm()) {
        if e > norm {
            norm = e;
        }
    }
    norm
}

/// Calculate the induced L-1 norm of the complex matrix.
///
/// The induced L-1 norm of a matrix is defined as
/// the maximum sum of the norm of the elements of its column vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::induced_l1_matrix_norm, matrix::Matrix},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 2, 2>::of(
///         &[(1.0, 1.0), (2.0, 0.0), (3.0, 0.0), (4.0, -1.0)]
///             .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let l1_norm = induced_l1_matrix_norm(&m);
///     assert_eq!(l1_norm, Float::of(17.0f32.sqrt() + 2.0));
/// }
/// ```
pub fn induced_l1_matrix_norm<F, const R: usize, const C: usize>(m: &Matrix<Complex<F>, R, C>) -> F
where
    F: RealFloat,
{
    let mut norm = F::zero();
    for e in (1..=C).map(|p| {
        m.get_column(p)
            .map(|c| {
                c.linear_iter()
                    .cloned()
                    .map(|e| e.clone().norm())
                    .fold(F::zero(), |acc, e| acc + e)
            })
            .expect(concat!(
                "Error[matrix::complex::induced_l1_matrix_norm]: ",
                "Failed to retrieve column vectors of the matrix."
            ))
    }) {
        if e > norm {
            norm = e;
        }
    }
    norm
}

/// Calculate the induced L-inf norm of the complex matrix.
///
/// The induced L-inf norm of a matrix is defined as
/// the maximum sum of the norm of the elements of its row vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::induced_l_inf_matrix_norm, matrix::Matrix},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 2, 2>::of(
///         &[(1.0, 1.0), (2.0, 0.0), (3.0, 0.0), (4.0, -1.0)]
///             .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let l_inf_norm = induced_l_inf_matrix_norm(&m);
///     assert_eq!(l_inf_norm, Float::of(17.0f32.sqrt() + 3.0));
/// }
/// ```
pub fn induced_l_inf_matrix_norm<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> F
where
    F: RealFloat,
{
    let mut norm = F::zero();
    for e in (1..=R).map(|p| {
        m.get_row(p)
            .map(|r| {
                r.linear_iter()
                    .cloned()
                    .map(|e| e.clone().norm())
                    .fold(F::zero(), |acc, e| acc + e)
            })
            .expect(concat!(
                "Error[matrix::complex::induced_l_inf_matrix_norm]: ",
                "Failed to retrieve row vectors of the matrix."
            ))
    }) {
        if e > norm {
            norm = e;
        }
    }
    norm
}

#[doc(cfg(feature = "extra"))]
/// Calculate the induced L-2 norm of the complex matrix.
///
/// The induced L-2 norm of a matrix is defined as
/// the square root of the spectral radius of the product
/// of the matrix and its conjugate transpose.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::induced_l2_matrix_norm, matrix::Matrix},
///     number::instances::{complex::Complex, double::Double},
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Double>, 2, 2>::of(
///         &[(1.0, 2.0), (3.0, -1.0), (4.0, 5.0), (6.0, -3.0)]
///             .map(|(r, i)| Complex::of(Double::of(r), Double::of(i))),
///     )
///     .unwrap();
///     let l2_norm = induced_l2_matrix_norm(&m);
///     assert_eq!(
///         l2_norm,
///         Double::of(((101.0 + 10085.0f64.sqrt()) / 2.0).sqrt())
///     );
/// }
/// ```
#[cfg(feature = "extra")]
pub fn induced_l2_matrix_norm<F, const R: usize, const C: usize>(m: &Matrix<Complex<F>, R, C>) -> F
where
    F: RealFloat,
{
    let conjugate_transposed = conjugate_transpose(m);
    let e = if R < C {
        extra::eigen_system_power(&apply(&(m.clone() * conjugate_transposed), |c| c.norm())).0
    } else {
        extra::eigen_system_power(&apply(&(conjugate_transposed * m.clone()), |c| c.norm())).0
    };
    e.absolute_value().square_root()
}
