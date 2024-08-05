//! # Utils
//!
//! some tools for matrix

use crate::error::Error;
use crate::error::Result;
use crate::matrix::Matrix;
use crate::number::Fractional;
use crate::number::Number;
use crate::vector::l2_norm_c;
use crate::vector::times_d;
use crate::vector::VectorC;

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
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # use rmatrix_ks::utils::horizontal_concat;
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
/// # use rmatrix_ks::utils::vertical_concat;
/// # use rmatrix_ks::error::Error;
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

/// transform the sqaure matrix to lower triangle form by rows elimination
pub(crate) fn lower_triangularize<T, const ROW: usize>(
    mat: Matrix<T, ROW, ROW>,
) -> Result<(Matrix<T, ROW, ROW>, Matrix<T, ROW, ROW>)>
where
    T: Number,
{
    let mut reduced = mat.to_owned();
    let mut p_all = Matrix::<T, ROW, ROW>::eyes()?;

    if !(is_lower_triangle_matrix(&reduced)? || ROW < 2) {
        let mut next: usize = 0;
        for index in (2..=ROW).rev() {
            // prevent out of boundary
            if index <= next {
                break;
            }
            // check pivot
            'check_pivot: while reduced.get_element(index, index - next)?.is_zero() {
                // find non-zero pivot
                for above in (1..=(index - 1)).rev() {
                    //do row exchange
                    if !reduced.get_element(above, index - next)?.is_zero() {
                        let p_change = Matrix::<T, ROW, ROW>::p_change(index, above)?;
                        p_all = p_change.to_owned().times(p_all)?;
                        reduced = p_change.times(reduced)?;
                        break 'check_pivot;
                    }
                }
                // find next column
                if index > next {
                    next = next + 1;
                }
            }
            // do eliminate
            let value = reduced.to_owned();
            let pivot = value.get_element(index, index - next)?;
            for over in (1..(index - 1)).rev() {
                // do row add
                let over_pivot = reduced.get_element(over, index - next)?;
                // skip zero line
                if !over_pivot.is_zero() {
                    // warn: for integer, division is non-accuracy, can use rational number
                    let factor = over_pivot.to_owned().ndiv(pivot.to_owned())?;
                    let p_add = Matrix::<T, ROW, ROW>::p_add(index, over, -factor)?;
                    p_all = p_add.to_owned().times(p_all)?;
                    reduced = p_add.times(reduced)?;
                }
            }
        }
    }
    Ok((p_all, reduced))
}

/// plu decomposition
///
/// all non-strange matrix can be decomposed into p, l, u,
/// which means p * l * u = m, and l is lower triangle matrix,
/// u is upper triangle matrix, p is permutation matrix
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::plu_decomposition;
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat: Matrix<f32, 2, 2> = Matrix::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
/// let plu = plu_decomposition(mat)?;
/// assert_eq!(Matrix::create(vec![1.0f32, 0.0f32, 0.0f32, 1.0f32])?, plu.0);
/// assert_eq!(Matrix::create(vec![1.0f32, 0.0f32, 3.0f32, 1.0f32])?, plu.1);
/// assert_eq!(Matrix::create(vec![1.0f32, 2.0f32, 0.0f32, -2.0f32])?, plu.2);
/// # Ok(())
/// # }
/// ```
pub fn plu_decomposition<T, const ROW: usize, const COL: usize>(
    mat: Matrix<T, ROW, COL>,
) -> Result<(
    Matrix<T, ROW, ROW>,
    Matrix<T, ROW, ROW>,
    Matrix<T, ROW, COL>,
)>
where
    T: Number,
{
    if mat.determinant()?.is_zero() {
        Err(Error::SingularMatrix)
    } else {
        let eliminates = mat.row_eliminate()?;
        let pl = lower_triangularize(eliminates.1.inverse()?)?;
        Ok((pl.0, pl.1, eliminates.0))
    }
}

/// qr decomposition
///
/// use Gram-Schmidt method
///
/// q * r = m
///
/// ```rust
/// # use rmatrix_ks::error::Result;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::qr_decomposition;
/// # fn main() -> Result<()> {
/// let mat = Matrix::<f32, 3, 3>::create(vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])?;
/// let qr = qr_decomposition(mat.to_owned())?;
/// assert!(qr.0.times(qr.1)?.equal(&mat));
/// # Ok(())
/// # }
/// ```
pub fn qr_decomposition<T, const ROW: usize, const COL: usize>(
    mat: Matrix<T, ROW, COL>,
) -> Result<(Matrix<T, ROW, COL>, Matrix<T, COL, COL>)>
where
    T: Fractional,
{
    let mut an = Vec::with_capacity(COL);
    for col in 1..=COL {
        an.push(mat.get_col(col)?);
    }
    let mut un: Vec<VectorC<T, ROW>> = Vec::with_capacity(COL);
    let mut en: Vec<VectorC<T, ROW>> = Vec::with_capacity(COL);
    for col in 1..=COL {
        // get a[i]
        let ai = match an.get(col - 1) {
            Some(element) => Ok(element),
            None => Err(Error::Message(format!(
                "read index {} out of boundary",
                col
            ))),
        }?
        .to_owned();

        #[cfg(feature = "rayon_mat")]
        let ai = VectorC::<T, ROW>::create(
            ai.inner
                .par_iter()
                .map(|e| e.to_owned().to_owned())
                .collect(),
        )?;

        #[cfg(not(feature = "rayon_mat"))]
        let ai =
            VectorC::<T, ROW>::create(ai.inner.iter().map(|e| e.to_owned().to_owned()).collect())?;

        // get u[i]
        let mut ui = ai.to_owned();
        for index in 1..=(col - 1) {
            let ek = match en.get(index - 1) {
                Some(element) => Ok(element),
                None => Err(Error::Message(format!(
                    "read index {} out of boundary",
                    index
                ))),
            }?
            .to_owned();

            #[cfg(feature = "rayon_mat")]
            let ek = VectorC::<T, ROW>::create(
                ek.inner
                    .par_iter()
                    .map(|e| e.to_owned().to_owned())
                    .collect(),
            )?;

            #[cfg(not(feature = "rayon_mat"))]
            let ek = VectorC::<T, ROW>::create(
                ek.inner.iter().map(|e| e.to_owned().to_owned()).collect(),
            )?;

            ui = ui.subtract(ek.to_owned().muls(times_d(ai.to_owned(), ek)?)?)?;
        }
        un.push(ui.to_owned());
        let norm = l2_norm_c(ui.to_owned());
        let ei = ui.divs(norm)?;
        en.push(ei);
    }
    let mut q = Matrix::<T, ROW, COL>::zeros()?;
    let mut r = Matrix::<T, COL, COL>::zeros()?;

    for row in 1..=ROW {
        for col in 1..=COL {
            let ei = match en.get(col - 1) {
                Some(element) => Ok(element),
                None => Err(Error::Message(format!(
                    "read index {} out of boundary",
                    col
                ))),
            }?
            .to_owned();
            q.set_element(row, col, ei.get_element(row, 1)?.to_owned().to_owned())?;
        }
    }

    for col1 in 1..=COL {
        let ei = match en.get(col1 - 1) {
            Some(element) => Ok(element),
            None => Err(Error::Message(format!(
                "read index {} out of boundary",
                col1
            ))),
        }?
        .to_owned();

        #[cfg(feature = "rayon_mat")]
        let ei = VectorC::<T, ROW>::create(
            ei.inner
                .par_iter()
                .map(|e| e.to_owned().to_owned())
                .collect(),
        )?;

        #[cfg(not(feature = "rayon_mat"))]
        let ei =
            VectorC::<T, ROW>::create(ei.inner.iter().map(|e| e.to_owned().to_owned()).collect())?;

        for col2 in col1..=COL {
            let ai = match an.get(col2 - 1) {
                Some(element) => Ok(element),
                None => Err(Error::Message(format!(
                    "read index {} out of boundary",
                    col2
                ))),
            }?
            .to_owned();

            #[cfg(feature = "rayon_mat")]
            let ai = VectorC::<T, ROW>::create(
                ai.inner
                    .par_iter()
                    .map(|e| e.to_owned().to_owned())
                    .collect(),
            )?;

            #[cfg(not(feature = "rayon_mat"))]
            let ai = VectorC::<T, ROW>::create(
                ai.inner.iter().map(|e| e.to_owned().to_owned()).collect(),
            )?;

            r.set_element(col1, col2, times_d(ai, ei.to_owned())?)?;
        }
    }

    Ok((q, r))
}

/// eigen values
///
/// suggest iter number is 1024
///
/// ```rust
/// # use rmatrix_ks::error::Result;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::number::Zero;
/// # use rmatrix_ks::utils::eigen_values;
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
/// only square matrix have the only solution
///
/// **FIXME**
///
/// ## Warning
///
/// the return type is not the perfact shape (c, e)
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::linear_solve;
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
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
) -> Result<Matrix<T, ROW, EDGE>>
where
    T: Number,
{
    if mat.rank()? > COL {
        Err(Error::NoSolution(ROW, COL))
    } else {
        mat.row_reduce()?.1.times(b)
    }
}

/// eigen vectors
///
/// **TODO**
pub fn eigen_system() {
    todo!()
}

/// predicate whether a matrix is ​​square
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::is_sqaure_matrix;
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
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
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<f32, 2, 3> = Matrix::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
/// let mat2: Matrix<f32, 2, 2> = Matrix::create(vec![1.0f32, 2.0f32, 0.0f32, 4.0f32])?;
/// assert_eq!(false, is_upper_triangle_matrix(&mat1)?);
/// assert!(is_upper_triangle_matrix(&mat2)?);
/// # Ok(())
/// # }
/// ```
pub fn is_upper_triangle_matrix<T, const ROW: usize, const COL: usize>(
    mat: &Matrix<T, ROW, COL>,
) -> Result<bool>
where
    T: Number,
{
    #[cfg(feature = "rayon_mat")]
    let predicate = points(|r, c| (r, c), ROW, COL)
        .par_iter()
        .filter(|(r, c)| r > c)
        .all(|(r, c)| {
            mat.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        });

    #[cfg(not(feature = "rayon_mat"))]
    let predicate = points(|r, c| (r, c), ROW, COL)
        .iter()
        .filter(|(r, c)| r > c)
        .all(|(r, c)| {
            mat.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        });

    Ok(predicate)
}

/// predicate whether a matrix is lower triangle
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::is_lower_triangle_matrix;
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<f32, 2, 3> = Matrix::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
/// let mat2: Matrix<f32, 2, 2> = Matrix::create(vec![1.0f32, 0.0f32, 2.0f32, 4.0f32])?;
/// assert_eq!(false, is_lower_triangle_matrix(&mat1)?);
/// assert!(is_lower_triangle_matrix(&mat2)?);
/// # Ok(())
/// # }
/// ```
pub fn is_lower_triangle_matrix<T, const ROW: usize, const COL: usize>(
    mat: &Matrix<T, ROW, COL>,
) -> Result<bool>
where
    T: Number,
{
    #[cfg(feature = "rayon_mat")]
    let predicate = points(|r, c| (r, c), ROW, COL)
        .par_iter()
        .filter(|(r, c)| r < c)
        .all(|(r, c)| {
            mat.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        });

    #[cfg(not(feature = "rayon_mat"))]
    let predicate = points(|r, c| (r, c), ROW, COL)
        .iter()
        .filter(|(r, c)| r < c)
        .all(|(r, c)| {
            mat.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        });

    Ok(predicate)
}

/// predicate whether a matrix is diagonal
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::is_diagonal_matrix;
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<f32, 2, 3> = Matrix::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
/// let mat2: Matrix<f32, 2, 2> = Matrix::create(vec![1.0f32, 0.0f32, 0.0f32, 4.0f32])?;
/// assert_eq!(false, is_diagonal_matrix(&mat1)?);
/// assert!(is_diagonal_matrix(&mat2)?);
/// # Ok(())
/// # }
/// ```
pub fn is_diagonal_matrix<T, const ROW: usize, const COL: usize>(
    mat: &Matrix<T, ROW, COL>,
) -> Result<bool>
where
    T: Number,
{
    #[cfg(feature = "rayon_mat")]
    let predicate = points(|r, c| (r, c), ROW, COL)
        .par_iter()
        .filter(|(r, c)| r != c)
        .all(|(r, c)| {
            mat.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        });

    #[cfg(not(feature = "rayon_mat"))]
    let predicate = points(|r, c| (r, c), ROW, COL)
        .iter()
        .filter(|(r, c)| r != c)
        .all(|(r, c)| {
            mat.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        });

    Ok(predicate)
}

/// predicate whether a matrix is identity
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::is_identity_matrix;
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<f32, 2, 3> = Matrix::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
/// let mat2: Matrix<f32, 2, 2> = Matrix::create(vec![1.0f32, 0.0f32, 0.0f32, 1.0f32])?;
/// assert_eq!(false, is_identity_matrix(&mat1)?);
/// assert!(is_identity_matrix(&mat2)?);
/// # Ok(())
/// # }
/// ```
pub fn is_identity_matrix<T, const ROW: usize, const COL: usize>(
    mat: &Matrix<T, ROW, COL>,
) -> Result<bool>
where
    T: Number,
{
    #[cfg(feature = "rayon_mat")]
    let predicate = points(|r, c| (r, c), ROW, COL)
        .par_iter()
        .filter(|(r, c)| r != c)
        .all(|(r, c)| {
            mat.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        })
        && points(|r, c| (r, c), ROW, COL)
            .par_iter()
            .filter(|(r, c)| r == c)
            .all(|(r, c)| {
                mat.get_element(r.to_owned(), c.to_owned())
                    .is_ok_and(|e| e.is_one())
            });

    #[cfg(not(feature = "rayon_mat"))]
    let predicate = points(|r, c| (r, c), ROW, COL)
        .iter()
        .filter(|(r, c)| r != c)
        .all(|(r, c)| {
            mat.get_element(r.to_owned(), c.to_owned())
                .is_ok_and(|e| e.is_zero())
        })
        && points(|r, c| (r, c), ROW, COL)
            .iter()
            .filter(|(r, c)| r == c)
            .all(|(r, c)| {
                mat.get_element(r.to_owned(), c.to_owned())
                    .is_ok_and(|e| e.is_one())
            });

    Ok(predicate)
}

/// predicate whether a matrix is orthogonal
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::is_orthogonal_matrix;
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat1: Matrix<f32, 2, 3> = Matrix::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
/// let mat2: Matrix<f32, 2, 2> =
///     Matrix::create(vec![1.0, -1.0, 1.0, 1.0])?.muls(1.0 / 2.0f32.sqrt())?;
/// assert_eq!(false, is_orthogonal_matrix(&mat1)?);
/// assert!(is_orthogonal_matrix(&mat2)?);
/// # Ok(())
/// # }
/// ```
pub fn is_orthogonal_matrix<T, const ROW: usize, const COL: usize>(
    mat: &Matrix<T, ROW, COL>,
) -> Result<bool>
where
    T: Number,
{
    let transposed = mat.transpose()?;

    if ROW > COL {
        is_identity_matrix::<T, COL, COL>(&transposed.times(mat.to_owned())?)
    } else {
        is_identity_matrix::<T, ROW, ROW>(&mat.to_owned().times(transposed)?)
    }
}

/// predicate whether a matrix is orthogonal
///
/// ```rust
/// # use rmatrix_ks::cmplx;
/// # use rmatrix_ks::complex::Complex;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::is_unitary_matrix;
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let mat: Matrix<Complex<f64>, 2, 2> =
///     Matrix::create(vec![
///         cmplx!(1.0, 0.0), cmplx!(0.0, 1.0),
///         cmplx!(0.0, 1.0), cmplx!(1.0, 0.0)])?
///     .divs(cmplx!(2.0f64.sqrt(), 0.0))?;
/// assert!(is_unitary_matrix(&mat)?);
/// # Ok(())
/// # }
/// ```
pub fn is_unitary_matrix<T, const ROW: usize, const COL: usize>(
    mat: &Matrix<T, ROW, COL>,
) -> Result<bool>
where
    T: Number,
{
    let transposed = mat.conjugate_transpose()?;

    if ROW > COL {
        is_identity_matrix::<T, COL, COL>(&transposed.times(mat.to_owned())?)
    } else {
        is_identity_matrix::<T, ROW, ROW>(&mat.to_owned().times(transposed)?)
    }
}
