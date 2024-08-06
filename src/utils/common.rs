//! # Common
//!
//! common tools for matrix or vectors

use crate::error::Result;
use crate::matrix::Matrix;
use crate::num::number::Fractional;
use crate::num::number::Number;
use crate::utils::decompose::qr_decomposition;

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
/// let mat1: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
/// let mat2: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
/// assert_eq!(Matrix::create(vec![1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6])?,
///     horizontal_concat(&mat1, &mat2)?);
/// # Ok(())
/// # }
/// ```
pub fn horizontal_concat<T, const ROW: usize, const COL: usize, const RCOL: usize>(
    mat: &Matrix<T, ROW, COL>,
    rhs: &Matrix<T, ROW, RCOL>,
) -> Result<Matrix<T, ROW, { COL + RCOL }>>
where
    T: Clone + Default + std::marker::Send + std::marker::Sync,
{
    let mut hmat = Matrix::zeros()?;
    for r in 1..=ROW {
        for c1 in 1..=COL {
            hmat.set_element(r, c1, mat.get_element(r, c1)?.to_owned())?;
        }

        for c2 in 1..=RCOL {
            hmat.set_element(r, COL + c2, rhs.get_element(r, c2)?.to_owned())?;
        }
    }
    Ok(hmat)
}

/// concatenate two matrices vertically
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::common::vertical_concat;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
/// let mat2: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
/// assert_eq!(Matrix::create(vec![1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6])?,
///     vertical_concat(&mat1, &mat2)?);
/// # Ok(())
/// # }
/// ```
pub fn vertical_concat<T, const ROW: usize, const COL: usize, const RROW: usize>(
    mat: &Matrix<T, ROW, COL>,
    rhs: &Matrix<T, RROW, COL>,
) -> Result<Matrix<T, { ROW + RROW }, COL>>
where
    T: Clone + std::marker::Send + std::marker::Sync,
{
    Matrix::create([&mat.inner[..], &rhs.inner[..]].concat())
}

/// eigen values
///
/// suggest iter number is 1024
///
/// ```rust
/// # use rmatrix_ks::error::Result;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::num::number::Zero;
/// # use rmatrix_ks::utils::common::eigen_values;
/// # fn main() -> Result<()> {
/// let mat = Matrix::<f32, 2, 2>::create(vec![0.0, 2.0, 2.0, 3.0])?;
/// let eigens = eigen_values(mat, 10)?;
/// assert!(eigens
///     .iter()
///     .zip(&[4.0, -1.0])
///     .all(|(r, n)| (r - n).is_zero()));
/// # Ok(())
/// # }
/// ```
pub fn eigen_values<T, const ROW: usize>(
    mat: Matrix<T, ROW, ROW>,
    iter_count: usize,
) -> Result<Vec<T>>
where
    T: Fractional,
{
    let qr = qr_decomposition(mat.to_owned())?;
    let mut a = qr.1.times(qr.0)?;
    for _ in 0..iter_count {
        let qrn = qr_decomposition(a.to_owned())?;
        a = qrn.1.times(qrn.0)?;
    }
    let mut eigens = Vec::with_capacity(ROW);
    for index in 1..=ROW {
        eigens.push(a.get_element(index, index)?.to_owned());
    }
    Ok(eigens)
}

/// solve linear equations
///
/// **TODO**
pub fn linear_solve<T, const ROW: usize, const COL: usize, const EDGE: usize>(
    _mat: Matrix<T, ROW, COL>,
    _b: Matrix<T, ROW, EDGE>,
) -> Result<Matrix<T, COL, EDGE>>
where
    T: Number,
{
    todo!()
}

/// eigen vectors
///
/// **TODO**
pub fn eigen_system() {
    todo!()
}
