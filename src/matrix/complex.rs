//! # matrix::complex
//!
//! Functions for handling vectors and matrices with complex elements.

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    matrix::{
        DEFAULT_MAX_ITER,
        math::{
            is_diagonal_matrix,
            is_identity_matrix,
            is_square_matrix,
            is_upper_triangular_matrix,
        },
        matrix::Matrix,
        utils::{apply, null_space, points_2d, transpose},
        vector::{VectorC, layer_product},
    },
    number::{
        instances::{complex::Complex, word8::Word8},
        traits::{
            floating::Floating,
            fractional::Fractional,
            number::Number,
            one::One,
            realfloat::RealFloat,
            zero::Zero,
        },
        utils::from_integral,
    },
};

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
pub fn is_unitary_matrix<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> bool
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
pub fn is_hermitian_matrix<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> bool
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
///         dot_product(&v1, &v2),
///         Complex::of(Float::of(1.0), Float::of(-5.0))
///     )
/// }
/// ```
pub fn dot_product<F, const R: usize>(
    v1: &VectorC<Complex<F>, R>,
    v2: &VectorC<Complex<F>, R>,
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

/// Project one complex vector onto another complex vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::project_to, vector::VectorC},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let v1 = VectorC::<Complex<Float>, 2>::of(
///         &[(1.0, 2.0), (3.0, -1.0)].map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let v2 = VectorC::<Complex<Float>, 2>::of(
///         &[(2.0, 1.0), (1.0, -3.0)].map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let p = project_to(&v1, &v2);
///     let p_expect = VectorC::<Complex<Float>, 2>::of(
///         &[(3.0 / 5.0, 32.0 / 15.0), (43.0 / 15.0, -19.0 / 15.0)]
///             .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     assert_eq!(p, p_expect);
/// }
/// ```
pub fn project_to<F, const R: usize>(
    from: &VectorC<Complex<F>, R>,
    to: &VectorC<Complex<F>, R>,
) -> VectorC<Complex<F>, R>
where
    F: RealFloat,
{
    let p1 = dot_product(to, from);
    let p2 = dot_product(to, to);
    to.clone() * (p1 / p2)
}

/// Apply the Gram-Schmidt process to the given basis
/// to obtain the corresponding orthogonal basis.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         complex::{dot_product, gram_schmidt_process},
///         matrix::Matrix,
///         utils::apply,
///     },
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::{one::One, zero::Zero},
///     },
/// };
///
/// fn main() {
///     let basis = Matrix::<Complex<Float>, 3, 3>::of(
///         &[
///             (1.0, 0.0),
///             (-1.0, 0.0),
///             (0.0, 0.0),
///             (0.0, 0.0),
///             (0.0, 1.0),
///             (-1.0, 0.0),
///             (0.0, 1.0),
///             (1.0, 0.0),
///             (1.0, 1.0),
///         ]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let ob = gram_schmidt_process(&basis);
///     let c1 = apply(&ob.get_column(1).unwrap(), |e: &Complex<Float>| e.clone());
///     let c2 = apply(&ob.get_column(2).unwrap(), |e: &Complex<Float>| e.clone());
///     let c3 = apply(&ob.get_column(3).unwrap(), |e: &Complex<Float>| e.clone());
///     // Each column vector is normalized.
///     assert_eq!(
///         (
///             dot_product(&c1, &c1),
///             dot_product(&c2, &c2),
///             dot_product(&c3, &c3),
///         ),
///         (Complex::one(), Complex::one(), Complex::one())
///     );
///     // The column vectors are mutually orthogonal.
///     assert_eq!(
///         (
///             dot_product(&c1, &c2),
///             dot_product(&c2, &c3),
///             dot_product(&c3, &c1),
///         ),
///         (Complex::zero(), Complex::zero(), Complex::zero())
///     );
///     // Corresponding orthogonal basis.
///     let ob_expect = Matrix::<Complex<Float>, 3, 3>::of(
///         &[
///             (1.0 / 2.0f32.sqrt(), 0.0),
///             (-1.0 / 8.0f32.sqrt(), 1.0 / 8.0f32.sqrt()),
///             (0.0, 0.5),
///             (0.0, 0.0),
///             (0.0, 1.0 / 2.0f32.sqrt()),
///             (-0.5, -0.5),
///             (0.0, 1.0 / 2.0f32.sqrt()),
///             (1.0 / 8.0f32.sqrt(), 1.0 / 8.0f32.sqrt()),
///             (0.5, 0.0),
///         ]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     assert_eq!(ob, ob_expect);
/// }
/// ```
pub fn gram_schmidt_process<F, const R: usize, const C: usize>(
    basis: &Matrix<Complex<F>, R, C>,
) -> Matrix<Complex<F>, R, C>
where
    F: RealFloat,
{
    let mut orthonormal_basis = Matrix::default();
    for k in 1..=C {
        // uk1 = bk
        // where Basis = [b1 | b2 | ... | bc]
        let mut uk = apply(
            &basis.get_column(k).expect(concat!(
                "Error[matrix::complex::gram_schmidt_process]: ",
                "Failed to retrieve the column vector of basis."
            )),
            |e: &Complex<F>| e.clone(),
        );
        // uk = bk - sum(proj(bk, uj), (j, 1, k - 1))
        for j in 1..k {
            // Use MGS, ukj = uk(j - 1) - proj(uk(j - 1), uj)
            let uj = apply(
                &orthonormal_basis.get_column(j).expect(concat!(
                    "Error[matrix::complex::gram_schmidt_process]: ",
                    "Failed to retrieve the column vector of orthonormal_basis."
                )),
                |e: &Complex<F>| e.clone(),
            );
            uk = uk.clone() - project_to(&uk, &uj);
        }
        // Normalize uk.
        uk = normalize(&uk);
        for row in 1..=R {
            // OB = [u1 | u2 | ... | uc]
            orthonormal_basis[(row, k)] = uk[(row, 1)].clone();
        }
    }
    orthonormal_basis
}

/// Generate the corresponding complex Givens rotation matrix
/// to eliminate the element at '(row, column)' using '(row - 1, column)'.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::givens_rotation_matrix, matrix::Matrix},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 2, 2>::of(
///         &[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
///             .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let g = givens_rotation_matrix(&m, 2, 1);
///     let g_expect = Matrix::<Complex<Float>, 2, 2>::of(
///         &[
///             (1.0 / 10.0f32.sqrt(), 0.0),
///             (3.0 / 10.0f32.sqrt(), 0.0),
///             (-3.0 / 10.0f32.sqrt(), 0.0),
///             (1.0 / 10.0f32.sqrt(), 0.0),
///         ]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     assert_eq!(g, g_expect);
/// }
/// ```
pub fn givens_rotation_matrix<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
    row: usize,
    column: usize,
) -> Matrix<Complex<F>, R, R>
where
    F: RealFloat,
{
    let mut givens = Matrix::<Complex<F>, R, R>::eyes();
    // e1 is the element to be eliminated.
    let e1 = m[(row, column)].clone();
    // e2 is the element used for elimination.
    let e2 = m[(row - 1, column)].clone();
    // v = (e2, e1), r = ||v||
    let r = (e1.clone().conjugate() * e1.clone() + e2.clone().conjugate() * e2.clone())
        .square_root();
    // sin(theta) = y / r = e1 / r
    let sin_theta = e1 / r.clone();
    // cos(theta) = x / r = e2 / r
    let cos_theta = e2 / r;
    // rotation {{c.conj, s.conj}, {-s, c}}
    givens[(row, row)] = cos_theta.clone();
    givens[(row - 1, row - 1)] = cos_theta.conjugate();
    givens[(row, row - 1)] = -sin_theta.clone();
    givens[(row - 1, row)] = sin_theta.conjugate();
    givens
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

/// Normalize the complex vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::normalize, vector::VectorC},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::{floating::Floating, zero::Zero},
///     },
/// };
///
/// fn main() {
///     let v1 = VectorC::<Complex<Float>, 2>::of(
///         &[(1.0, 0.0), (0.0, -1.0)].map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let normalized = normalize(&v1);
///     let normalized_expect = v1 / Complex::of(Float::of(2.0).square_root(), Float::zero());
///     assert_eq!(normalized, normalized_expect);
/// }
/// ```
pub fn normalize<F, const R: usize>(v: &VectorC<Complex<F>, R>) -> VectorC<Complex<F>, R>
where
    F: RealFloat,
{
    if v.inner.par_iter().all(|e| e.is_zero()) {
        VectorC::default()
    } else {
        let v_norm = euclidean_norm(v);
        v.clone() / v_norm
    }
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
pub fn induced_l1_matrix_norm<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> F
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

// **TODO** L-2 norm

/// Compute the QR decomposition of a complex matrix using the Gram-Schmidt process.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         complex::{is_unitary_matrix, qr_decomposition_gs},
///         math::is_upper_triangular_matrix,
///         matrix::Matrix,
///     },
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 3, 3>::of(
///         &[
///             (1.0, 1.0),
///             (2.0, 0.0),
///             (3.0, 0.0),
///             (4.0, 0.0),
///             (5.0, 2.0),
///             (6.0, 0.0),
///             (7.0, 0.0),
///             (8.0, 0.0),
///             (1.0, 3.0),
///         ]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_gs(&m).unwrap();
///     assert!(is_unitary_matrix(&q));
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
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::qr_decomposition_gs, matrix::Matrix},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 2, 3>::of(
///         &[
///             (1.0, 1.0),
///             (2.0, 0.0),
///             (3.0, 0.0),
///             (4.0, 0.0),
///             (5.0, 2.0),
///             (6.0, 0.0),
///         ]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let qr = qr_decomposition_gs(&m);
///     assert_eq!(qr, None);
/// }
/// ```
///
/// </div>
pub fn qr_decomposition_gs<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> Option<(Matrix<Complex<F>, R, C>, Matrix<Complex<F>, C, C>)>
where
    F: RealFloat,
{
    if R < C {
        eprintln!(concat!(
            "Error[matrix::extra::qr_decomposition_gs]: ",
            "The matrix should be a tall matrix or a square matrix"
        ));
        None
    } else {
        // Q = MGS(M)
        let q = gram_schmidt_process(m);
        // R = Upper {  a_c . e_r } and R = Q^T A
        let r = conjugate_transpose(&q) * m.clone();
        Some((q, r))
    }
}

/// Compute the QR decomposition of a complex matrix using Householder transformations.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         complex::{is_unitary_matrix, qr_decomposition_h},
///         math::is_upper_triangular_matrix,
///         matrix::Matrix,
///     },
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 3, 3>::of(
///         &[
///             (1.0, 1.0),
///             (2.0, 0.0),
///             (3.0, 0.0),
///             (4.0, 0.0),
///             (5.0, 2.0),
///             (6.0, 0.0),
///             (7.0, 0.0),
///             (8.0, 0.0),
///             (1.0, 3.0),
///         ]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_h(&m);
///     assert!(is_unitary_matrix(&q));
///     assert!(is_upper_triangular_matrix(&r));
///     assert_eq!(q * r, m);
/// }
/// ```
pub fn qr_decomposition_h<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> (Matrix<Complex<F>, R, R>, Matrix<Complex<F>, R, C>)
where
    F: RealFloat,
{
    let mut q = Matrix::<Complex<F>, R, R>::eyes();
    let mut r = m.clone();
    let two = Complex::one() + Complex::one();
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
        // Forall ei in x where i < c is zero
        for index in 1..column {
            x[(index, 1)] = Complex::zero();
        }
        let x_norm = euclidean_norm(&x);
        // v = x but v[col] = x[col] + x_norm * signum(x[col])
        let mut v = x.clone();
        v[(column, 1)] = x[(column, 1)].clone() + x_norm * x[(column, 1)].sign_number();
        // p = I - 2 / (v^H . v) (v * v^H)
        let p = Matrix::<Complex<F>, R, R>::eyes()
            - layer_product(&v, &conjugate_transpose(&v)) * two.clone() / dot_product(&v, &v);
        // r = pn p_{n - 1} ... p1 m
        r = p.clone() * r;
        // q = p1 p2 ... pn
        q = q * p;
    }
    (q, r)
}

/// Compute the QR decomposition of a complex matrix using Givens rotation matrices.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         complex::{is_unitary_matrix, qr_decomposition_gr},
///         math::is_upper_triangular_matrix,
///         matrix::Matrix,
///     },
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 3, 3>::of(
///         &[
///             (1.0, 1.0),
///             (2.0, 0.0),
///             (3.0, 0.0),
///             (4.0, 0.0),
///             (5.0, 2.0),
///             (6.0, 0.0),
///             (7.0, 0.0),
///             (8.0, 0.0),
///             (1.0, 3.0),
///         ]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_gr(&m);
///     assert!(is_unitary_matrix(&q));
///     assert!(is_upper_triangular_matrix(&r));
///     assert_eq!(q * r, m);
/// }
/// ```
pub fn qr_decomposition_gr<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> (Matrix<Complex<F>, R, R>, Matrix<Complex<F>, R, C>)
where
    F: RealFloat,
{
    let mut q = Matrix::<Complex<F>, R, R>::eyes();
    let mut r = m.clone();
    for column in 1..=C {
        for row in (column..R).rev() {
            if r[(row + 1, column)].is_zero() {
                continue;
            } else {
                let rotation = givens_rotation_matrix(&r, row + 1, column);
                r = rotation.clone() * r;
                q = rotation * q;
            }
        }
    }
    (conjugate_transpose(&q), r)
}

/// Compute the economy-sized QR decomposition of a complex matrix using Givens rotation matrices.
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
///     matrix::{
///         complex::{is_unitary_matrix, qr_decomposition_es},
///         matrix::Matrix,
///     },
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 3, 2>::of(
///         &[
///             (1.0, 1.0),
///             (2.0, 0.0),
///             (3.0, 0.0),
///             (4.0, 2.0),
///             (5.0, 0.0),
///             (6.0, 0.0),
///         ]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let (q, r) = qr_decomposition_es(&m);
///     let q_expect = Matrix::<Complex<Float>, 3, 2>::of(
///         &[
///             (1.0 / 6.0, 1.0 / 6.0),
///             (4.0 / 117.0f32.sqrt(), -2.0 / 13.0f32.sqrt()),
///             (0.5, 0.0),
///             (1.0 / 52.0f32.sqrt(), 5.0 / 52.0f32.sqrt()),
///             (5.0 / 6.0, 0.0),
///             (-1.0 / 468.0f32.sqrt(), -5.0 / 468.0f32.sqrt()),
///         ]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     assert!(is_unitary_matrix(&q));
///     assert_eq!(q, q_expect);
///     let r_expect = Matrix::<Complex<Float>, 2, 2>::of(
///         &[
///             (6.0, 0.0),
///             (22.0 / 3.0, 2.0 / 3.0),
///             (0.0, 0.0),
///             (52.0f32.sqrt() / 3.0, 0.0),
///         ]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     assert_eq!(r, r_expect);
/// }
/// ```
pub fn qr_decomposition_es<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> (
    Matrix<Complex<F>, R, { Matrix::<Complex<F>, R, C>::get_diagonal_length() }>,
    Matrix<Complex<F>, { Matrix::<Complex<F>, R, C>::get_diagonal_length() }, C>,
)
where
    F: RealFloat,
    [(); Matrix::<Complex<F>, R, C>::get_diagonal_length()]:,
{
    let (basic_q, basic_r) = qr_decomposition_gr(m);
    let thin = Matrix::<Complex<F>, R, C>::get_diagonal_length();
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

/// Decompose the complex matrix into
/// an orthogonal matrix and the corresponding Hessenberg matrix.
///
/// # Examples
///
/// ```rust
/// use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
/// use rmatrix_ks::{
///     matrix::{
///         complex::{conjugate_transpose, hessenberg_decomposition, is_unitary_matrix},
///         matrix::Matrix,
///         utils::points_2d,
///     },
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 4, 4>::of(
///         &[
///             (1.0, 1.0),
///             (2.0, 0.0),
///             (3.0, 0.0),
///             (4.0, 0.0),
///             (5.0, 0.0),
///             (6.0, 1.0),
///             (7.0, 0.0),
///             (8.0, 0.0),
///             (9.0, 0.0),
///             (10.0, 0.0),
///             (11.0, 1.0),
///             (12.0, 0.0),
///             (13.0, 0.0),
///             (14.0, 0.0),
///             (15.0, 0.0),
///             (16.0, 0.0),
///         ]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let (p, h) = hessenberg_decomposition(&m);
///     assert!(
///         points_2d((1, 4), (1, 4), |r, c| r > c + 1)
///             .par_iter()
///             .all(|&p| h[p].is_zero())
///     );
///     assert!(is_unitary_matrix(&p));
///     // P . H . P^H = M
///     assert_eq!(p.clone() * h * conjugate_transpose(&p), m);
/// }
/// ```
pub fn hessenberg_decomposition<F, const E: usize>(
    m: &Matrix<Complex<F>, E, E>,
) -> (Matrix<Complex<F>, E, E>, Matrix<Complex<F>, E, E>)
where
    F: RealFloat,
{
    if E < 3 {
        (Matrix::eyes(), m.clone())
    } else {
        let mut hessen = m.clone();
        let mut p = Matrix::eyes();
        for column in 1..=E {
            for row in (column + 1..E).rev() {
                let pi = conjugate_transpose(&givens_rotation_matrix(&hessen, row + 1, column));
                hessen = conjugate_transpose(&pi) * hessen * pi.clone();
                p = p * pi;
            }
        }
        (p, hessen)
    }
}

/// Calculate the eigenvalues and eigenvectors of the complex matrix
/// using the QR algorithm based on Givens rotation matrices.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::eigen_system_qr, matrix::Matrix, vector::VectorC},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::{one::One, zero::Zero},
///     },
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 2, 2>::of(&[
///         Complex::zero(),
///         -Complex::unit_i(),
///         Complex::unit_i(),
///         Complex::zero(),
///     ])
///     .unwrap();
///     let (es, evs) = eigen_system_qr(&m, 1024);
///     assert_eq!(
///         es,
///         VectorC::<Complex<Float>, 2>::of(&[Complex::one(), -Complex::one()]).unwrap()
///     );
///     let evs_expect = Matrix::<Complex<Float>, 2, 2>::of(&[
///         -Complex::unit_i(),
///         Complex::unit_i(),
///         Complex::one(),
///         Complex::one(),
///     ])
///     .unwrap();
///     assert_eq!(evs, evs_expect);
/// }
/// ```
pub fn eigen_system_qr<F, const E: usize>(
    m: &Matrix<Complex<F>, E, E>,
    max_iter: usize,
) -> (VectorC<Complex<F>, E>, Matrix<Complex<F>, E, E>)
where
    F: RealFloat,
{
    let mut eigenvalues = VectorC::default();
    if is_diagonal_matrix(m) {
        for idx in 1..=E {
            eigenvalues[(idx, 1)] = m[(idx, idx)].clone();
        }
        (eigenvalues, Matrix::eyes())
    } else {
        if E < 3 {
            let four = from_integral::<Complex<F>, Word8>(Word8::of(4));
            let b = -m[(1, 1)].clone() - m[(2, 2)].clone();
            let c =
                m[(1, 1)].clone() * m[(2, 2)].clone() - m[(1, 2)].clone() * m[(2, 1)].clone();
            let delta = b.clone() * b.clone() - four * c;
            eigenvalues[(1, 1)] = (-b.clone() + delta.clone().square_root()) * Complex::half();
            eigenvalues[(2, 1)] = (-b - delta.square_root()) * Complex::half();
        } else {
            let (_, mut hm) = hessenberg_decomposition(m);
            for _ in 0..max_iter {
                // delta = (a1 - a2) / 2
                let delta = (hm[(E - 1, E - 1)].clone() - hm[(E, E)].clone()) * Complex::half();
                // shift = a2 + delta - sign(delta) * sqrt(delta * delta + b1 * b2)
                // [[a1, b1], [b2, a2]]
                let wilkinson = hm[(E, E)].clone() + delta.clone()
                    - delta.sign_number()
                        * (delta.clone() * delta
                            + hm[(E - 1, E)].clone() * hm[(E, E - 1)].clone())
                        .square_root();
                let (q, r) = qr_decomposition_gr(&(hm - Matrix::eyes() * wilkinson.clone()));
                hm = r * q + Matrix::eyes() * wilkinson;
                if is_upper_triangular_matrix(&hm) {
                    break;
                }
            }
            for idx in 1..=E {
                eigenvalues[(idx, 1)] = hm[(idx, idx)].clone();
            }
        }
        let mut eigenvectors = Matrix::default();
        for idx in 1..=E {
            let eigen_equation = m.clone() - Matrix::eyes() * eigenvalues[(idx, 1)].clone();
            let null_space_eq = null_space(&eigen_equation);
            let eigenvector = if null_space_eq.is_empty() {
                let ev = gram_schmidt_process(&eigen_equation);
                apply(
                    &ev.get_column(idx).expect(&format!(
                        concat!(
                            "Error[matrix::extra::eigen_system_qr]: ",
                            "Failed to obtain the {}-th column vector of ev."
                        ),
                        idx
                    )),
                    |e: &Complex<F>| e.clone(),
                )
            } else {
                null_space_eq[0].clone()
            };
            for row in 1..=E {
                eigenvectors[(row, idx)] = eigenvector[(row, 1)].clone();
            }
        }
        (eigenvalues, eigenvectors)
    }
}

/// Calculate the induced L-2 norm of the matrix.
///
/// aka. spectral norm.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::induced_l2_matrix_norm, matrix::Matrix},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let m = Matrix::<Complex<Float>, 3, 3>::of(
///         &[
///             (1.0, -2.0),
///             (3.0, -4.0),
///             (5.0, 6.0),
///             (7.0, -8.0),
///             (9.0, 10.0),
///             (11.0, -12.0),
///             (13.0, 14.0),
///             (15.0, -16.0),
///             (17.0, 18.0),
///         ]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
///     )
///     .unwrap();
///     let n = induced_l2_matrix_norm(&m);
///     assert_eq!(n, Float::of(40.2086));
/// }
/// ```
pub fn induced_l2_matrix_norm<F, const R: usize, const C: usize>(
    m: &Matrix<Complex<F>, R, C>,
) -> F
where
    F: RealFloat,
{
    let p = conjugate_transpose(m) * m.clone();
    let (rho_square, _) = eigen_system_qr(&p, DEFAULT_MAX_ITER);
    rho_square[(1, 1)].real.clone().square_root()
}
