//! # matrix::complex
//!
//! Functions for handling vectors and matrices with complex elements.

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    matrix::{
        DEFAULT_MAX_ITER,
        Matrix,
        math::{is_diagonal_matrix, is_identity_matrix, is_square_matrix, is_upper_triangular_matrix},
        utils::{apply, null_space, points_2d, transpose},
        vector::{is_column_vector, is_row_vector, layer_product},
    },
    number::{
        instances::{complex::Complex, word8::Word8},
        traits::{floating::Floating, fractional::Fractional, number::Number, one::One, realfloat::RealFloat, zero::Zero},
        utils::from_integral,
    },
};

/// Obtains the conjugate transpose of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, complex::conjugate_transpose},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[(1.0, 2.0), (3.0, -1.0), (4.0, 0.0), (5.0, 6.0)]
///         .map(|(real, imag)| Complex::of(Float::of(real), Float::of(imag))),
/// )
/// .unwrap();
/// let n = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[(1.0, -2.0), (4.0, 0.0), (3.0, 1.0), (5.0, -6.0)]
///         .map(|(real, imag)| Complex::of(Float::of(real), Float::of(imag))),
/// )
/// .unwrap();
/// assert_eq!(conjugate_transpose(&m), n);
/// ```
pub fn conjugate_transpose<F>(m: &Matrix<Complex<F>>) -> Matrix<Complex<F>>
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
///     matrix::{Matrix, complex::is_unitary_matrix},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// let m1 = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[
///         Complex::of(Float::of(1.0f32 / 2.0f32.sqrt()), Float::zero()),
///         Complex::of(Float::of(-1.0f32 / 2.0f32.sqrt()), Float::zero()),
///         Complex::of(Float::of(1.0f32 / 2.0f32.sqrt()), Float::zero()),
///         Complex::of(Float::of(1.0f32 / 2.0f32.sqrt()), Float::zero()),
///     ],
/// )
/// .unwrap();
/// let m2 = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[
///         Complex::of(Float::of(1.0f32), Float::zero()),
///         Complex::of(Float::of(-1.0), Float::zero()),
///         Complex::of(Float::zero(), Float::zero()),
///         Complex::of(Float::of(1.0f32), Float::zero()),
///     ],
/// )
/// .unwrap();
/// assert!(is_unitary_matrix(&m1));
/// assert!(!is_unitary_matrix(&m2));
/// ```
pub fn is_unitary_matrix<F>(m: &Matrix<Complex<F>>) -> bool
where
    F: RealFloat,
{
    let conjugate_transposed = conjugate_transpose(m);
    if m.row < m.column {
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
///     matrix::{Matrix, complex::is_normal_matrix},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// let m1 = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[
///         Complex::of(Float::of(1.0f32), Float::zero()),
///         Complex::of(Float::of(2.0f32), Float::zero()),
///         Complex::of(Float::of(2.0f32), Float::zero()),
///         Complex::of(Float::of(3.0f32), Float::zero()),
///     ],
/// )
/// .unwrap();
/// let m2 = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[
///         Complex::of(Float::of(1.0f32), Float::zero()),
///         Complex::of(Float::of(1.0f32), Float::zero()),
///         Complex::of(Float::of(0.0f32), Float::zero()),
///         Complex::of(Float::of(1.0f32), Float::zero()),
///     ],
/// )
/// .unwrap();
/// assert!(is_normal_matrix(&m1));
/// assert!(!is_normal_matrix(&m2));
/// ```
pub fn is_normal_matrix<F>(m: &Matrix<Complex<F>>) -> bool
where
    F: RealFloat,
{
    let conjugate_transposed = conjugate_transpose(m);
    is_square_matrix(m) && (conjugate_transposed.clone() * m.clone() == m.clone() * conjugate_transposed)
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
///     matrix::{Matrix, complex::is_hermitian_matrix},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// let m1 = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[
///         Complex::of(Float::of(2.0f32), Float::zero()),
///         Complex::of(Float::of(1.0f32), Float::of(1.0f32)),
///         Complex::of(Float::of(1.0f32), Float::of(-1.0f32)),
///         Complex::of(Float::of(3.0f32), Float::zero()),
///     ],
/// )
/// .unwrap();
/// let m2 = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[
///         Complex::of(Float::of(1.0f32), Float::zero()),
///         Complex::of(Float::of(2.0f32), Float::of(1.0f32)),
///         Complex::of(Float::of(3.0f32), Float::zero()),
///         Complex::of(Float::of(4.0f32), Float::zero()),
///     ],
/// )
/// .unwrap();
/// assert!(is_hermitian_matrix(&m1));
/// assert!(!is_hermitian_matrix(&m2));
/// ```
pub fn is_hermitian_matrix<F>(m: &Matrix<Complex<F>>) -> bool
where
    F: RealFloat,
{
    is_square_matrix(m)
        && points_2d((1, m.row), (1, m.column), |row, col| row <= col)
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
///     matrix::{Matrix, complex::is_anti_hermitian_matrix},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// let m1 = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[
///         Complex::of(Float::zero(), Float::zero()),
///         Complex::of(Float::of(1.0f32), Float::of(1.0f32)),
///         Complex::of(Float::of(-1.0f32), Float::of(1.0f32)),
///         Complex::of(Float::zero(), Float::zero()),
///     ],
/// )
/// .unwrap();
/// let m2 = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[
///         Complex::of(Float::of(1.0f32), Float::zero()),
///         Complex::of(Float::of(2.0f32), Float::of(1.0f32)),
///         Complex::of(Float::of(3.0f32), Float::zero()),
///         Complex::of(Float::of(4.0f32), Float::zero()),
///     ],
/// )
/// .unwrap();
/// assert!(is_anti_hermitian_matrix(&m1));
/// assert!(!is_anti_hermitian_matrix(&m2));
/// ```
pub fn is_anti_hermitian_matrix<F>(m: &Matrix<Complex<F>>) -> bool
where
    F: RealFloat,
{
    is_square_matrix(m)
        && points_2d((1, m.row), (1, m.column), |row, col| row <= col)
            .par_iter()
            .all(|&p @ (row, col)| m[p] == -m[(col, row)].clone().conjugate())
}

/// Calculate the dot product of a complex vector.
///
/// # Panics
///
/// The dimensions of the column vectors involved in the function must match.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::dot_product, vector::column_vector},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let v1 = column_vector::<Complex<Float>>(
///     4,
///     &[(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let v2 = column_vector::<Complex<Float>>(
///     4,
///     &[(3.0, -4.0), (6.0, -2.0), (1.0, 2.0), (4.0, 3.0)]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// assert_eq!(
///     dot_product(&v1, &v2),
///     Complex::of(Float::of(1.0), Float::of(-5.0))
/// )
/// ```
pub fn dot_product<F>(v1: &Matrix<Complex<F>>, v2: &Matrix<Complex<F>>) -> Complex<F>
where
    F: RealFloat,
{
    assert!(
        is_column_vector(v1) && is_column_vector(v2) && v1.row == v2.row,
        concat!(
            "Error[matrix::complex::dot_product]: ",
            "Only column vectors with matching dimensions can be multiplied."
        )
    );

    v1.inner
        .par_iter()
        .zip(v2.inner.par_iter())
        .map(|(e1, e2)| e1.clone().conjugate() * e2.clone())
        .reduce(Complex::<F>::zero, |acc, e| acc + e)
}

/// Project one complex vector onto another complex vector.
///
/// # Panics
///
/// This function can only be used for column vectors with matching dimensions.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::project_to, vector::column_vector},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let v1 = column_vector::<Complex<Float>>(
///     2,
///     &[(1.0, 2.0), (3.0, -1.0)].map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let v2 = column_vector::<Complex<Float>>(
///     2,
///     &[(2.0, 1.0), (1.0, -3.0)].map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let p = project_to(&v1, &v2);
/// let p_expect = column_vector::<Complex<Float>>(
///     2,
///     &[(3.0 / 5.0, 32.0 / 15.0), (43.0 / 15.0, -19.0 / 15.0)]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// assert_eq!(p, p_expect);
/// ```
pub fn project_to<F>(from: &Matrix<Complex<F>>, to: &Matrix<Complex<F>>) -> Matrix<Complex<F>>
where
    F: RealFloat,
{
    assert!(
        is_column_vector(from) && is_column_vector(to) && from.row == to.row,
        concat!(
            "Error[matrix::complex::project_to]: ",
            "This function can only be used for column vectors with matching dimensions."
        )
    );

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
///         Matrix,
///         complex::{dot_product, gram_schmidt_process},
///         utils::apply,
///     },
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::{one::One, zero::Zero},
///     },
/// };
///
/// let basis = Matrix::<Complex<Float>>::of(
///     3,
///     3,
///     &[
///         (1.0, 0.0),
///         (-1.0, 0.0),
///         (0.0, 0.0),
///         (0.0, 0.0),
///         (0.0, 1.0),
///         (-1.0, 0.0),
///         (0.0, 1.0),
///         (1.0, 0.0),
///         (1.0, 1.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let ob = gram_schmidt_process(&basis);
/// let c1 = apply(&ob.get_column(1).unwrap(), |e: &Complex<Float>| e.clone());
/// let c2 = apply(&ob.get_column(2).unwrap(), |e: &Complex<Float>| e.clone());
/// let c3 = apply(&ob.get_column(3).unwrap(), |e: &Complex<Float>| e.clone());
/// // Each column vector is normalized.
/// assert_eq!(
///     (
///         dot_product(&c1, &c1),
///         dot_product(&c2, &c2),
///         dot_product(&c3, &c3),
///     ),
///     (Complex::one(), Complex::one(), Complex::one())
/// );
/// // The column vectors are mutually orthogonal.
/// assert_eq!(
///     (
///         dot_product(&c1, &c2),
///         dot_product(&c2, &c3),
///         dot_product(&c3, &c1),
///     ),
///     (Complex::zero(), Complex::zero(), Complex::zero())
/// );
/// // Corresponding orthogonal basis.
/// let ob_expect = Matrix::<Complex<Float>>::of(
///     3,
///     3,
///     &[
///         (1.0 / 2.0f32.sqrt(), 0.0),
///         (-1.0 / 8.0f32.sqrt(), 1.0 / 8.0f32.sqrt()),
///         (0.0, 0.5),
///         (0.0, 0.0),
///         (0.0, 1.0 / 2.0f32.sqrt()),
///         (-0.5, -0.5),
///         (0.0, 1.0 / 2.0f32.sqrt()),
///         (1.0 / 8.0f32.sqrt(), 1.0 / 8.0f32.sqrt()),
///         (0.5, 0.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// assert_eq!(ob, ob_expect);
/// ```
pub fn gram_schmidt_process<F>(basis: &Matrix<Complex<F>>) -> Matrix<Complex<F>>
where
    F: RealFloat,
{
    let mut orthonormal_basis = Matrix::defaults(basis.row, basis.column);
    for k in 1..=basis.column {
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
        for row in 1..=basis.row {
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
///     matrix::{Matrix, complex::givens_rotation_matrix},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let g = givens_rotation_matrix(&m, 2, 1);
/// let g_expect = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[
///         (1.0 / 10.0f32.sqrt(), 0.0),
///         (3.0 / 10.0f32.sqrt(), 0.0),
///         (-3.0 / 10.0f32.sqrt(), 0.0),
///         (1.0 / 10.0f32.sqrt(), 0.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// assert_eq!(g, g_expect);
/// ```
pub fn givens_rotation_matrix<F>(m: &Matrix<Complex<F>>, row: usize, column: usize) -> Matrix<Complex<F>>
where
    F: RealFloat,
{
    let mut givens = Matrix::<Complex<F>>::eyes(m.row, m.row);
    // e1 is the element to be eliminated.
    let e1 = m[(row, column)].clone();
    // e2 is the element used for elimination.
    let e2 = m[(row - 1, column)].clone();
    // v = (e2, e1), r = ||v||
    let r = (e1.clone().conjugate() * e1.clone() + e2.clone().conjugate() * e2.clone()).square_root();
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
/// # Panics
///
/// Only vectors have the Euclidean norm.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::euclidean_norm, vector::column_vector},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// let v = column_vector::<Complex<Float>>(
///     2,
///     &[(1.0, 7.0), (2.0, -6.0)].map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// assert_eq!(
///     euclidean_norm(&v),
///     Complex::of(Float::of(3.0 * 10.0f32.sqrt()), Float::zero())
/// )
/// ```
pub fn euclidean_norm<F>(v: &Matrix<Complex<F>>) -> Complex<F>
where
    F: RealFloat,
{
    assert!(
        is_row_vector(v) || is_column_vector(v),
        concat!(
            "Error[matrix::complex::euclidean_norm]: ",
            "Only vectors have the Euclidean norm."
        )
    );

    v.inner
        .par_iter()
        .map(|e| e.clone().conjugate() * e.clone())
        .reduce(Complex::<F>::zero, |acc, e| acc + e)
        .square_root()
}

/// Normalize the complex vector.
///
/// # Panics
///
/// This function can only be used for column vector normalization.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::normalize, vector::column_vector},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::{floating::Floating, zero::Zero},
///     },
/// };
///
/// let v1 = column_vector::<Complex<Float>>(
///     2,
///     &[(1.0, 0.0), (0.0, -1.0)].map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let normalized = normalize(&v1);
/// let normalized_expect = v1 / Complex::of(Float::of(2.0).square_root(), Float::zero());
/// assert_eq!(normalized, normalized_expect);
/// ```
pub fn normalize<F>(v: &Matrix<Complex<F>>) -> Matrix<Complex<F>>
where
    F: RealFloat,
{
    assert!(
        is_column_vector(v),
        concat!(
            "Error[matrix::vector::normalize]: ",
            "This function can only be used for column vector normalization."
        )
    );

    if v.inner.par_iter().all(|e| e.is_zero()) {
        Matrix::defaults(v.row, 1)
    } else {
        let v_norm = euclidean_norm(v);
        v.clone() / v_norm
    }
}

/// Calculate the maximum norm for complex vector.
///
/// Aka L_inf-norm.
///
/// # Panics
///
/// This function can only be used for column vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{complex::maximum_norm, vector::column_vector},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let m = column_vector::<Complex<Float>>(
///     3,
///     &[(1.0, 1.0), (-2.0, 0.0), (3.0, -4.0)]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let l_inf_norm = maximum_norm(&m);
/// assert_eq!(l_inf_norm, Float::of(5.0));
/// ```
pub fn maximum_norm<F>(v: &Matrix<Complex<F>>) -> F
where
    F: RealFloat,
{
    assert!(
        is_column_vector(v),
        concat!(
            "Error[matrix::vector::maximum_norm]: ",
            "This function can only be used for column vectors."
        )
    );

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
///     matrix::{Matrix, complex::induced_l1_matrix_norm},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[(1.0, 1.0), (2.0, 0.0), (3.0, 0.0), (4.0, -1.0)]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let l1_norm = induced_l1_matrix_norm(&m);
/// assert_eq!(l1_norm, Float::of(17.0f32.sqrt() + 2.0));
/// ```
pub fn induced_l1_matrix_norm<F>(m: &Matrix<Complex<F>>) -> F
where
    F: RealFloat,
{
    let mut norm = F::zero();
    for e in (1..=m.column).map(|p| {
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
///     matrix::{Matrix, complex::induced_l_inf_matrix_norm},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[(1.0, 1.0), (2.0, 0.0), (3.0, 0.0), (4.0, -1.0)]
///         .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let l_inf_norm = induced_l_inf_matrix_norm(&m);
/// assert_eq!(l_inf_norm, Float::of(17.0f32.sqrt() + 3.0));
/// ```
pub fn induced_l_inf_matrix_norm<F>(m: &Matrix<Complex<F>>) -> F
where
    F: RealFloat,
{
    let mut norm = F::zero();
    for e in (1..=m.row).map(|p| {
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

/// Compute the QR decomposition of a complex matrix using the Gram-Schmidt process.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         Matrix,
///         complex::{is_unitary_matrix, qr_decomposition_gs},
///         math::is_upper_triangular_matrix,
///     },
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     3,
///     3,
///     &[
///         (1.0, 1.0),
///         (2.0, 0.0),
///         (3.0, 0.0),
///         (4.0, 0.0),
///         (5.0, 2.0),
///         (6.0, 0.0),
///         (7.0, 0.0),
///         (8.0, 0.0),
///         (1.0, 3.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let (q, r) = qr_decomposition_gs(&m).unwrap();
/// assert!(is_unitary_matrix(&q));
/// assert!(is_upper_triangular_matrix(&r));
/// assert_eq!(q * r, m);
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
///     matrix::{Matrix, complex::qr_decomposition_gs},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     2,
///     3,
///     &[
///         (1.0, 1.0),
///         (2.0, 0.0),
///         (3.0, 0.0),
///         (4.0, 0.0),
///         (5.0, 2.0),
///         (6.0, 0.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let qr = qr_decomposition_gs(&m);
/// assert_eq!(qr, None);
/// ```
///
/// </div>
pub fn qr_decomposition_gs<F>(m: &Matrix<Complex<F>>) -> Option<(Matrix<Complex<F>>, Matrix<Complex<F>>)>
where
    F: RealFloat,
{
    if m.row < m.column {
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
///         Matrix,
///         complex::{is_unitary_matrix, qr_decomposition_h},
///         math::is_upper_triangular_matrix,
///     },
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     3,
///     3,
///     &[
///         (1.0, 1.0),
///         (2.0, 0.0),
///         (3.0, 0.0),
///         (4.0, 0.0),
///         (5.0, 2.0),
///         (6.0, 0.0),
///         (7.0, 0.0),
///         (8.0, 0.0),
///         (1.0, 3.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let (q, r) = qr_decomposition_h(&m);
/// assert!(is_unitary_matrix(&q));
/// assert!(is_upper_triangular_matrix(&r));
/// assert_eq!(q * r, m);
/// ```
pub fn qr_decomposition_h<F>(m: &Matrix<Complex<F>>) -> (Matrix<Complex<F>>, Matrix<Complex<F>>)
where
    F: RealFloat,
{
    let mut q = Matrix::<Complex<F>>::eyes(m.row, m.row);
    let mut r = m.clone();
    let two = Complex::one() + Complex::one();
    for column in 1..=m.edge() {
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
        let p = Matrix::<Complex<F>>::eyes(m.row, m.row)
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
///         Matrix,
///         complex::{is_unitary_matrix, qr_decomposition_gr},
///         math::is_upper_triangular_matrix,
///     },
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     3,
///     3,
///     &[
///         (1.0, 1.0),
///         (2.0, 0.0),
///         (3.0, 0.0),
///         (4.0, 0.0),
///         (5.0, 2.0),
///         (6.0, 0.0),
///         (7.0, 0.0),
///         (8.0, 0.0),
///         (1.0, 3.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let (q, r) = qr_decomposition_gr(&m);
/// assert!(is_unitary_matrix(&q));
/// assert!(is_upper_triangular_matrix(&r));
/// assert_eq!(q * r, m);
/// ```
pub fn qr_decomposition_gr<F>(m: &Matrix<Complex<F>>) -> (Matrix<Complex<F>>, Matrix<Complex<F>>)
where
    F: RealFloat,
{
    let mut q = Matrix::<Complex<F>>::eyes(m.row, m.row);
    let mut r = m.clone();
    for column in 1..=m.column {
        for row in (column..m.row).rev() {
            if r[(row + 1, column)].is_zero() {
                continue;
            } else if !r[(row + 1, column)].is_zero() {
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
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         Matrix,
///         complex::{is_unitary_matrix, qr_decomposition_es},
///     },
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     3,
///     2,
///     &[
///         (1.0, 1.0),
///         (2.0, 0.0),
///         (3.0, 0.0),
///         (4.0, 2.0),
///         (5.0, 0.0),
///         (6.0, 0.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let (q, r) = qr_decomposition_es(&m);
/// let q_expect = Matrix::<Complex<Float>>::of(
///     3,
///     2,
///     &[
///         (1.0 / 6.0, 1.0 / 6.0),
///         (4.0 / 117.0f32.sqrt(), -2.0 / 13.0f32.sqrt()),
///         (0.5, 0.0),
///         (1.0 / 52.0f32.sqrt(), 5.0 / 52.0f32.sqrt()),
///         (5.0 / 6.0, 0.0),
///         (-1.0 / 468.0f32.sqrt(), -5.0 / 468.0f32.sqrt()),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// assert!(is_unitary_matrix(&q));
/// assert_eq!(q, q_expect);
/// let r_expect = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[
///         (6.0, 0.0),
///         (22.0 / 3.0, 2.0 / 3.0),
///         (0.0, 0.0),
///         (52.0f32.sqrt() / 3.0, 0.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// assert_eq!(r, r_expect);
/// ```
pub fn qr_decomposition_es<F>(m: &Matrix<Complex<F>>) -> (Matrix<Complex<F>>, Matrix<Complex<F>>)
where
    F: RealFloat,
{
    let (basic_q, basic_r) = qr_decomposition_gr(m);
    let thin = m.edge();
    let mut q = Matrix::defaults(m.row, thin);
    let mut r = Matrix::defaults(thin, m.column);
    if m.row > m.column {
        // If m > n, then qr computes only the first n columns of Q and the first n rows of R.
        for row in 1..=m.row {
            for column in 1..=thin {
                q[(row, column)] = basic_q[(row, column)].clone();
            }
        }
        r.inner = basic_r.inner[..(thin * m.column)].to_vec();
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
/// # Panics
///
/// Only square matrices can undergo the Heisenberg decomposition.
///
/// # Examples
///
/// ```rust
/// use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
/// use rmatrix_ks::{
///     matrix::{
///         Matrix,
///         complex::{conjugate_transpose, hessenberg_decomposition, is_unitary_matrix},
///         utils::points_2d,
///     },
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     4,
///     4,
///     &[
///         (1.0, 1.0),
///         (2.0, 0.0),
///         (3.0, 0.0),
///         (4.0, 0.0),
///         (5.0, 0.0),
///         (6.0, 1.0),
///         (7.0, 0.0),
///         (8.0, 0.0),
///         (9.0, 0.0),
///         (10.0, 0.0),
///         (11.0, 1.0),
///         (12.0, 0.0),
///         (13.0, 0.0),
///         (14.0, 0.0),
///         (15.0, 0.0),
///         (16.0, 0.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let (p, h) = hessenberg_decomposition(&m);
/// assert!(
///     points_2d((1, 4), (1, 4), |r, c| r > c + 1)
///         .par_iter()
///         .all(|&pi| h[pi].is_zero())
/// );
/// assert!(is_unitary_matrix(&p));
/// // P . H . P^H = M
/// assert_eq!(p.clone() * h * conjugate_transpose(&p), m);
/// ```
pub fn hessenberg_decomposition<F>(m: &Matrix<Complex<F>>) -> (Matrix<Complex<F>>, Matrix<Complex<F>>)
where
    F: RealFloat,
{
    assert!(
        is_square_matrix(m),
        concat!(
            "Error[matrix::complex::hessenberg_decomposition]: ",
            "Only square matrices can undergo the Heisenberg decomposition."
        )
    );

    if m.edge() < 3 {
        (Matrix::eyes(m.row, m.column), m.clone())
    } else {
        let mut hessen = m.clone();
        let mut p = Matrix::eyes(m.row, m.column);
        for column in 1..=m.edge() {
            for row in (column + 1..m.edge()).rev() {
                if !hessen[(row + 1, column)].is_zero() {
                    let pi = conjugate_transpose(&givens_rotation_matrix(&hessen, row + 1, column));
                    hessen = conjugate_transpose(&pi) * hessen * pi.clone();
                    p = p * pi;
                }
            }
        }
        (p, hessen)
    }
}

/// Calculate the eigenvalues and eigenvectors of the complex matrix
/// using the QR algorithm based on Givens rotation matrices.
///
/// # Panics
///
/// Only square matrices can potentially have eigenvalues.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, complex::eigen_system_qr, vector::column_vector},
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::{one::One, zero::Zero},
///     },
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[
///         Complex::zero(),
///         -Complex::unit_i(),
///         Complex::unit_i(),
///         Complex::zero(),
///     ],
/// )
/// .unwrap();
/// let (es, evs) = eigen_system_qr(&m, 1024);
/// assert_eq!(
///     es,
///     column_vector::<Complex<Float>>(2, &[Complex::one(), -Complex::one()]).unwrap()
/// );
/// let evs_expect = Matrix::<Complex<Float>>::of(
///     2,
///     2,
///     &[
///         -Complex::of(Float::zero(), Float::of(1.0 / 2.0f32.sqrt())),
///         Complex::of(Float::zero(), Float::of(1.0 / 2.0f32.sqrt())),
///         Complex::of(Float::of(1.0 / 2.0f32.sqrt()), Float::zero()),
///         Complex::of(Float::of(1.0 / 2.0f32.sqrt()), Float::zero()),
///     ],
/// )
/// .unwrap();
/// assert_eq!(evs, evs_expect);
/// ```
pub fn eigen_system_qr<F>(m: &Matrix<Complex<F>>, max_iter: usize) -> (Matrix<Complex<F>>, Matrix<Complex<F>>)
where
    F: RealFloat,
{
    assert!(
        is_square_matrix(m),
        concat!(
            "Error[matrix::complex::eigen_system_qr]: ",
            "Only square matrices can potentially have eigenvalues."
        )
    );

    let mut eigenvalues = Matrix::defaults(m.edge(), 1);
    if is_diagonal_matrix(m) {
        for idx in 1..=m.edge() {
            eigenvalues[(idx, 1)] = m[(idx, idx)].clone();
        }
        (eigenvalues, Matrix::eyes(m.row, m.column))
    } else {
        if m.edge() < 3 {
            let four = from_integral::<Complex<F>, Word8>(Word8::of(4));
            let b = -m[(1, 1)].clone() - m[(2, 2)].clone();
            let c = m[(1, 1)].clone() * m[(2, 2)].clone() - m[(1, 2)].clone() * m[(2, 1)].clone();
            let delta = b.clone() * b.clone() - four * c;
            eigenvalues[(1, 1)] = (-b.clone() + delta.clone().square_root()) * Complex::half();
            eigenvalues[(2, 1)] = (-b - delta.square_root()) * Complex::half();
        } else {
            let (_, mut hm) = hessenberg_decomposition(m);
            for _ in 0..max_iter {
                // delta = (a1 - a2) / 2
                let delta = (hm[(m.edge() - 1, m.edge() - 1)].clone() - hm[(m.edge(), m.edge())].clone()) * Complex::half();
                // shift = a2 + delta - sign(delta) * sqrt(delta * delta + b1 * b2)
                // [[a1, b1], [b2, a2]]
                let wilkinson = hm[(m.edge(), m.edge())].clone() + delta.clone()
                    - delta.sign_number()
                        * (delta.clone() * delta + hm[(m.edge() - 1, m.edge())].clone() * hm[(m.edge(), m.edge() - 1)].clone())
                            .square_root();
                let (q, r) = qr_decomposition_gr(&(hm - Matrix::eyes(m.row, m.column) * wilkinson.clone()));
                hm = r * q + Matrix::eyes(m.row, m.column) * wilkinson;
                if is_upper_triangular_matrix(&hm) {
                    break;
                }
            }
            for idx in 1..=m.edge() {
                eigenvalues[(idx, 1)] = hm[(idx, idx)].clone();
            }
        }
        let mut eigenvectors = Matrix::defaults(m.row, m.column);
        for idx in 1..=m.edge() {
            let eigen_equation = m.clone() - Matrix::eyes(m.row, m.column) * eigenvalues[(idx, 1)].clone();
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
            let eigenvector = normalize(&eigenvector);
            for row in 1..=m.edge() {
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
///     matrix::{Matrix, complex::induced_l2_matrix_norm},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     3,
///     3,
///     &[
///         (1.0, -2.0),
///         (3.0, -4.0),
///         (5.0, 6.0),
///         (7.0, -8.0),
///         (9.0, 10.0),
///         (11.0, -12.0),
///         (13.0, 14.0),
///         (15.0, -16.0),
///         (17.0, 18.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let n = induced_l2_matrix_norm(&m);
/// assert_eq!(n, Float::of(40.2086));
/// ```
pub fn induced_l2_matrix_norm<F>(m: &Matrix<Complex<F>>) -> F
where
    F: RealFloat,
{
    let p = conjugate_transpose(m) * m.clone();
    let (rho_square, _) = eigen_system_qr(&p, DEFAULT_MAX_ITER);
    rho_square[(1, 1)].real.clone().square_root()
}

/// Compute the singular value decomposition of the complex matrix
///
/// The orthogonal basis part is based on the Gram-Schmidt process.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         Matrix,
///         complex::{conjugate_transpose, singular_value_decomposition},
///     },
///     number::{
///         instances::{complex::Complex, float::Float},
///         traits::zero::Zero,
///     },
/// };
///
/// let m = Matrix::<Complex<Float>>::of(
///     3,
///     3,
///     &[
///         (1.0, -2.0),
///         (3.0, -4.0),
///         (5.0, 6.0),
///         (7.0, -8.0),
///         (9.0, 10.0),
///         (11.0, -12.0),
///         (13.0, 14.0),
///         (15.0, -16.0),
///         (17.0, 18.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let (u, s, v) = singular_value_decomposition(&m);
/// let s_expect = Matrix::<Complex<Float>>::diagonal(
///     3,
///     3,
///     &[40.2086, 21.4557, 5.64978].map(|e| Complex::of(Float::of(e), Float::zero())),
/// )
/// .unwrap();
/// assert_eq!(s, s_expect);
/// assert_eq!(u * s * conjugate_transpose(&v), m);
/// ```
pub fn singular_value_decomposition<F>(m: &Matrix<Complex<F>>) -> (Matrix<Complex<F>>, Matrix<Complex<F>>, Matrix<Complex<F>>)
where
    F: RealFloat,
{
    let left_sym = m.clone() * conjugate_transpose(m);
    let right_sym = conjugate_transpose(m) * m.clone();
    let edge = m.edge();
    // Compute the eigenvectors and eigenvalues of the left matrix.
    let (sig1, u) = eigen_system_qr(&left_sym, DEFAULT_MAX_ITER);
    // Compute the eigenvectors and eigenvalues of the right matrix.
    let (sig2, v) = eigen_system_qr(&right_sym, DEFAULT_MAX_ITER);
    let mut sigma = Matrix::<Complex<F>>::defaults(m.row, m.column);
    for idx in 1..=edge {
        // Take the average to reduce the error.
        sigma[(idx, idx)] = if sig1[(idx, 1)].is_zero() || sig2[(idx, 1)].is_zero() {
            Complex::zero()
        } else {
            ((sig1[(idx, 1)].clone() + sig2[(idx, 1)].clone()) * Complex::half()).square_root()
        };
    }
    // Obtain the corresponding orthogonal basis.
    let mut u = gram_schmidt_process(&u);
    let mut v = gram_schmidt_process(&v);
    if m.row > m.column {
        // Use U as a reference to correct V.
        // A^H U = V (S^H) = V S'
        // => A^H ui = si vi
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
                |e: &Complex<F>| e.clone(),
            );
            let vi = if sigma[(idx, idx)].is_zero() {
                normalize(&(conjugate_transpose(m) * ui))
            } else {
                conjugate_transpose(m) * ui / sigma[(idx, idx)].clone()
            };
            for row in 1..=m.column {
                modified_v[(row, idx)] = vi[(row, 1)].clone();
            }
        }
        v = modified_v;
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
                |e: &Complex<F>| e.clone(),
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
            let p_left = Matrix::<Complex<F>>::p_change(m.row, m.row, idx, max);
            u = u * p_left;
            let p_right = Matrix::<Complex<F>>::p_change(m.column, m.column, idx, max);
            v = v * p_right;
        }
    }
    (u, sigma, v)
}

/// Compute the Moore-Penrose inverse of a complex matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         Matrix,
///         complex::{is_hermitian_matrix, moore_penrose_inverse},
///     },
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// // Properties that the Moore-Penrose inverse must satisfy.
/// let rect = Matrix::<Complex<Float>>::of(
///     2,
///     3,
///     &[
///         (1.0, 2.0),
///         (2.0, -1.0),
///         (3.0, 0.0),
///         (4.0, 0.0),
///         (5.0, 1.0),
///         (6.0, -2.0),
///     ]
///     .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
/// )
/// .unwrap();
/// let rectp = moore_penrose_inverse(&rect);
/// // A . Ap . A = A
/// assert_eq!(rect.clone() * rectp.clone() * rect.clone(), rect);
/// // Ap . A . Ap = Ap
/// assert_eq!(rectp.clone() * rect.clone() * rectp.clone(), rectp);
/// // A . Ap is a hermitian matrix.
/// assert!(is_hermitian_matrix(&(rect.clone() * rectp.clone())));
/// // Ap . A is also a hermitian matrix.
/// assert!(is_hermitian_matrix(&(rectp * rect)));
/// ```
pub fn moore_penrose_inverse<F>(m: &Matrix<Complex<F>>) -> Matrix<Complex<F>>
where
    F: RealFloat,
{
    let (u, s, v) = singular_value_decomposition(m);
    let mut sp = Matrix::defaults(s.column, s.row);
    for p in 1..=sp.edge() {
        if s[(p, p)].is_zero() {
            sp[(p, p)] = Complex::<F>::zero()
        } else {
            sp[(p, p)] = s[(p, p)].clone().reciprocal();
        }
    }
    v * sp * conjugate_transpose(&u)
}
