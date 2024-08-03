//! # Utils
//!
//! some tools for matrix

use crate::error::MatrixError;
use crate::matrix::Matrix;
use crate::number::Number;

#[cfg(feature = "rayon_mat")]
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// generate points of matrix
///
/// ```rust
/// # use rmatrix_ks::utils::points;
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
/// # use rmatrix_ks::error::MatrixError;
/// # use rmatrix_ks::utils::horizontal_concat;
/// # fn main() -> Result<(), MatrixError> {
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
) -> Result<Matrix<T, ROW, { COL + RCOL }>, MatrixError>
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
/// # use rmatrix_ks::utils::vertical_concat;
/// # use rmatrix_ks::error::MatrixError;
/// # fn main() -> Result<(), MatrixError> {
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
) -> Result<Matrix<T, { ROW + RROW }, COL>, MatrixError>
where
    T: Clone + std::marker::Send + std::marker::Sync,
{
    Matrix::create([&mat.inner[..], &rhs.inner[..]].concat())
}

/// solve linear equations
///
/// only square matrix have the only solution
///
/// ## Warning
///
/// the return type is not the perfact shape (c, e)
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::linear_solve;
/// # use rmatrix_ks::error::MatrixError;
/// # fn main() -> Result<(), MatrixError> {
/// let mat: Matrix<f32, 2, 2> = Matrix::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
/// let b: Matrix<f32, 2, 2> = Matrix::create(vec![5.0f32, 6.0f32, 7.0f32, 8.0f32])?;
/// assert_eq!(Matrix::create(vec![-3.0f32, -4.0f32, 4.0f32, 5.0f32])?,
///     linear_solve(mat, b)?);
/// # Ok(())
/// # }
/// ```
pub fn linear_solve<T, const ROW: usize, const COL: usize, const EDGE: usize>(
    mat: Matrix<T, ROW, COL>,
    b: Matrix<T, ROW, EDGE>,
) -> Result<Matrix<T, ROW, EDGE>, MatrixError>
where
    T: Number,
{
    if mat.rank()? > COL {
        Err(MatrixError::NoSolution(ROW, COL))
    } else {
        mat.row_reduce()?.1.times(b)
    }
}

pub fn lu_decomposition<T, const ROW: usize, const COL: usize>(
    mat: Matrix<T, ROW, COL>,
) -> Result<(Matrix<T, ROW, ROW>, Matrix<T, ROW, COL>), MatrixError>
where
    T: Number,
{
    let eliminates = mat.row_eliminate()?;
    Ok((eliminates.1.inverse()?, eliminates.0))
}

// qr decomposition
pub fn qr_decomposition() {
    todo!()
}

// eigen system
pub fn eigen_system() {
    todo!()
}

/// predicate whether a matrix is ​​square
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::is_sqaure_matrix;
/// # use rmatrix_ks::error::MatrixError;
/// # fn main() -> Result<(), MatrixError> {
/// let mat1: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
/// let mat2: Matrix<i8, 2, 2> = Matrix::create(vec![1, 2, 3, 4])?;
/// assert_eq!(false, is_sqaure_matrix(&mat1));
/// assert!(is_sqaure_matrix(&mat2));
/// # Ok(())
/// # }
/// ```
pub fn is_sqaure_matrix<T, const ROW: usize, const COL: usize>(_: &Matrix<T, ROW, COL>) -> bool {
    ROW == COL
}

/// predicate whether a matrix is upper triangle
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::is_upper_triangle_matrix;
/// # use rmatrix_ks::error::MatrixError;
/// # fn main() -> Result<(), MatrixError> {
/// let mat1: Matrix<f32, 2, 3> = Matrix::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
/// let mat2: Matrix<f32, 2, 2> = Matrix::create(vec![1.0f32, 2.0f32, 0.0f32, 4.0f32])?;
/// assert_eq!(false, is_upper_triangle_matrix(&mat1)?);
/// assert!(is_upper_triangle_matrix(&mat2)?);
/// # Ok(())
/// # }
/// ```
pub fn is_upper_triangle_matrix<T, const ROW: usize, const COL: usize>(
    m: &Matrix<T, ROW, COL>,
) -> Result<bool, MatrixError>
where
    T: Number,
{
    #[cfg(feature = "rayon_mat")]
    let predicate = points(|r, c| (r, c), ROW, COL)
        .par_iter()
        .filter(|(r, c)| r > c)
        .all(|(r, c)| {
            m.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        });

    #[cfg(not(feature = "rayon_mat"))]
    let predicate = points(|r, c| (r, c), ROW, COL)
        .iter()
        .filter(|(r, c)| r > c)
        .all(|(r, c)| {
            m.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        });

    Ok(predicate)
}
