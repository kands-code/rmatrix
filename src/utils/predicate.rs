//! # Predicate
//!
//! some predicate functions

use crate::matrix::Matrix;
use crate::num::number::{Number, One, Zero};
use crate::utils::common::points;
use crate::{error::Result, num::number::Equal};

/// predicate whether a matrix is ​​square
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::predicate::is_square_matrix;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<i8> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
/// let mat2: Matrix<i8> = Matrix::create(2, 2, vec![1, 2, 3, 4])?;
/// assert_eq!(false, is_square_matrix(&mat1));
/// assert!(is_square_matrix(&mat2));
/// # Ok(())
/// # }
/// ```
pub const fn is_square_matrix<T>(mat: &Matrix<T>) -> bool {
    mat.row() == mat.column()
}

/// predicate whether a matrix is symmetric
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::predicate::is_symmetric_matrix;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<i32> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
/// let mat2: Matrix<i32> = Matrix::create(2, 2, vec![1, 2, 2, 1])?;
/// assert_eq!(false, is_symmetric_matrix(&mat1)?);
/// assert!(is_symmetric_matrix(&mat2)?);
/// # Ok(())
/// # }
/// ```
pub fn is_symmetric_matrix<T>(mat: &Matrix<T>) -> Result<bool>
where
    T: Equal,
{
    if is_square_matrix(mat) {
        let mut is_symmetric = true;
        for row in 1..=mat.row() {
            for col in (row + 1)..=mat.row() {
                is_symmetric =
                    is_symmetric && mat.get_element(row, col)?.equal(mat.get_element(col, row)?);
            }
        }
        Ok(is_symmetric)
    } else {
        Ok(false)
    }
}

/// predicate whether a matrix is upper triangle
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::predicate::is_upper_triangle_matrix;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<f32> = Matrix::create(2, 3, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
/// let mat2: Matrix<f32> = Matrix::create(2, 2, vec![1.0f32, 2.0f32, 0.0f32, 4.0f32])?;
/// assert_eq!(false, is_upper_triangle_matrix(&mat1));
/// assert!(is_upper_triangle_matrix(&mat2));
/// # Ok(())
/// # }
/// ```
pub fn is_upper_triangle_matrix<T>(mat: &Matrix<T>) -> bool
where
    T: Zero,
{
    points(|r, c| (r, c), mat.row(), mat.column())
        .iter()
        .filter(|(r, c)| r > c)
        .all(|(r, c)| {
            mat.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        })
}

/// predicate whether a matrix is lower triangle
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::predicate::is_lower_triangle_matrix;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<f32> = Matrix::create(2, 3, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
/// let mat2: Matrix<f32> = Matrix::create(2, 2, vec![1.0f32, 0.0f32, 2.0f32, 4.0f32])?;
/// assert_eq!(false, is_lower_triangle_matrix(&mat1));
/// assert!(is_lower_triangle_matrix(&mat2));
/// # Ok(())
/// # }
/// ```
pub fn is_lower_triangle_matrix<T>(mat: &Matrix<T>) -> bool
where
    T: Zero,
{
    points(|r, c| (r, c), mat.row(), mat.column())
        .iter()
        .filter(|(r, c)| r < c)
        .all(|(r, c)| {
            mat.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        })
}

/// predicate whether a matrix is diagonal
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::predicate::is_diagonal_matrix;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<f32> = Matrix::create(2, 3, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
/// let mat2: Matrix<f32> = Matrix::create(2, 2, vec![1.0f32, 0.0f32, 0.0f32, 4.0f32])?;
/// assert_eq!(false, is_diagonal_matrix(&mat1));
/// assert!(is_diagonal_matrix(&mat2));
/// # Ok(())
/// # }
/// ```
pub fn is_diagonal_matrix<T>(mat: &Matrix<T>) -> bool
where
    T: Zero,
{
    points(|r, c| (r, c), mat.row(), mat.column())
        .iter()
        .filter(|(r, c)| r != c)
        .all(|(r, c)| {
            mat.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        })
}

/// predicate whether a matrix is identity
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::predicate::is_identity_matrix;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<f32> = Matrix::create(2, 3, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
/// let mat2: Matrix<f32> = Matrix::create(2, 2, vec![1.0f32, 0.0f32, 0.0f32, 1.0f32])?;
/// assert_eq!(false, is_identity_matrix(&mat1));
/// assert!(is_identity_matrix(&mat2));
/// # Ok(())
/// # }
/// ```
pub fn is_identity_matrix<T>(mat: &Matrix<T>) -> bool
where
    T: Zero + One,
{
    points(|r, c| (r, c), mat.row(), mat.column())
        .iter()
        .filter(|(r, c)| r != c)
        .all(|(r, c)| {
            mat.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        })
        && points(|r, c| (r, c), mat.row(), mat.column())
            .iter()
            .filter(|(r, c)| r == c)
            .all(|(r, c)| {
                mat.get_element(r.to_owned(), c.to_owned())
                    .is_ok_and(|e| e.is_one())
            })
}

/// predicate whether a matrix is orthogonal
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::predicate::is_orthogonal_matrix;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<f32> = Matrix::create(2, 3, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
/// let mat2: Matrix<f32> =
///     Matrix::create(2, 2, vec![1.0, -1.0, 1.0, 1.0])?.muls(1.0 / 2.0f32.sqrt())?;
/// assert_eq!(false, is_orthogonal_matrix(&mat1)?);
/// assert!(is_orthogonal_matrix(&mat2)?);
/// # Ok(())
/// # }
/// ```
pub fn is_orthogonal_matrix<T>(mat: &Matrix<T>) -> Result<bool>
where
    T: Number,
{
    let transposed = mat.transpose()?;

    if mat.row() > mat.column() {
        Ok(is_identity_matrix::<T>(&transposed.times(mat.to_owned())?))
    } else {
        Ok(is_identity_matrix::<T>(&mat.to_owned().times(transposed)?))
    }
}

/// predicate whether a matrix is orthogonal
///
/// ```rust
/// # use rmatrix_ks::cmplx;
/// # use rmatrix_ks::error::Result;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::num::complex::Complex;
/// # use rmatrix_ks::utils::predicate::is_unitary_matrix;
/// # fn main() -> Result<()> {
/// let mat: Matrix<Complex<f32>> =
///     Matrix::create(2, 2, vec![
///         cmplx!(1.0, 0.0), cmplx!(0.0, 1.0),
///         cmplx!(0.0, 1.0), cmplx!(1.0, 0.0)])?
///     .divs(cmplx!(2.0f32.sqrt(), 0.0))?;
/// assert!(is_unitary_matrix(&mat)?);
/// # Ok(())
/// # }
/// ```
pub fn is_unitary_matrix<T>(mat: &Matrix<T>) -> Result<bool>
where
    T: Number,
{
    let transposed = mat.conjugate_transpose()?;

    if mat.row() > mat.column() {
        Ok(is_identity_matrix(&transposed.times(mat.to_owned())?))
    } else {
        Ok(is_identity_matrix(&mat.to_owned().times(transposed)?))
    }
}
