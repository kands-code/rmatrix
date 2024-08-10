//! # Common
//!
//! common tools for matrix or vectors

use crate::error::{Error, Result};
use crate::matrix::Matrix;
use crate::num::number::Zero;

/// generate points of matrix
///
/// ```rust
/// # use rmatrix_ks::utils::common::points;
/// # fn main() {
/// assert_eq!(vec![(2, 3), (2, 6), (4, 3), (4, 6)],
///     points(|x, y| (2 * x, 3 * y), 2, 2));
/// # }
/// ```
pub fn points<T, R>(
    mut f: impl FnMut(usize, usize) -> (T, R),
    row: usize,
    col: usize,
) -> Vec<(T, R)> {
    let mut ps = Vec::with_capacity(row * col);
    for r in 1..=row {
        for c in 1..=col {
            ps.push(f(r, c))
        }
    }
    ps
}

/// concatenate two matrices horizontally
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::error::Result;
/// # use rmatrix_ks::utils::common::horizontal_concat;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<i32> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
/// let mat2: Matrix<i32> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
/// assert_eq!(Matrix::create(2, 6, vec![1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6])?,
///     horizontal_concat(&mat1, &mat2)?);
/// # Ok(())
/// # }
/// ```
pub fn horizontal_concat<T>(mat: &Matrix<T>, rhs: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Clone + Zero,
{
    if mat.row() != rhs.row() {
        Err(Error::IncompatibleShape((mat.row(), rhs.column()), rhs.dim))
    } else {
        let mut hmat = Matrix::zeros(mat.row(), mat.column() + rhs.column())?;
        for r in 1..=mat.row() {
            for c1 in 1..=mat.column() {
                hmat.set_element(r, c1, mat.get_element(r, c1)?.to_owned())?;
            }

            for c2 in 1..=rhs.column() {
                hmat.set_element(r, mat.column() + c2, rhs.get_element(r, c2)?.to_owned())?;
            }
        }
        Ok(hmat)
    }
}

/// concatenate two matrices vertically
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::common::vertical_concat;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<i32> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
/// let mat2: Matrix<i32> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
/// assert_eq!(Matrix::create(4, 3, vec![1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])?,
///     vertical_concat(&mat1, &mat2)?);
/// # Ok(())
/// # }
/// ```
pub fn vertical_concat<T>(mat: &Matrix<T>, rhs: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Clone,
{
    Matrix::create(
        mat.row() + rhs.row(),
        mat.column(),
        [&mat.inner[..], &rhs.inner[..]].concat(),
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
