//! # Decompose
//!
//! some decomposition method

use crate::error::IError;
use crate::error::IResult;
use crate::matrix::Matrix;
use crate::num::number::Fractional;
use crate::num::number::Number;
use crate::utils::predicate::is_upper_triangle_matrix;
use crate::vector::euclid_norm;
use crate::vector::identity_vector_column;
use crate::vector::times_d;
use crate::vector::times_v;
use crate::vector::ColumnVector;

/// plu decomposition
///
/// all non-strange matrix can be decomposed into p, l, u,
/// which means p * l * u = m, and l is lower triangle matrix,
/// u is upper triangle matrix, p is permutation matrix
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::decompose::plu_decomposition;
/// # use rmatrix_ks::error::IResult;
/// # fn main() -> IResult<()> {
/// let mat: Matrix<f32> = Matrix::create(2, 2, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
/// let plu = plu_decomposition(mat)?;
/// assert_eq!(Matrix::create(2, 2, vec![1.0f32, 0.0f32, 0.0f32, 1.0f32])?, plu.0);
/// assert_eq!(Matrix::create(2, 2, vec![1.0f32, 0.0f32, 3.0f32, 1.0f32])?, plu.1);
/// assert_eq!(Matrix::create(2, 2, vec![1.0f32, 2.0f32, 0.0f32, -2.0f32])?, plu.2);
/// # Ok(())
/// # }
/// ```
pub fn plu_decomposition<T>(mat: Matrix<T>) -> IResult<(Matrix<T>, Matrix<T>, Matrix<T>)>
where
    T: Number,
{
    let eliminates = mat.row_eliminate()?;
    if eliminates.2.get_diag()?.iter().any(|e| e.is_zero()) {
        Err(IError::SingularMatrix)
    } else {
        let (p, l, _, _) = eliminates.0.times(eliminates.1)?.row_eliminate()?;
        Ok((p, l, eliminates.2))
    }
}

/// qr decomposition
///
/// use Householder method
///
/// q^H * r = m
///
/// H is conjugate transpose
///
/// ```rust
/// # use rmatrix_ks::error::IResult;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::decompose::qr_decomposition;
/// # fn main() -> IResult<()> {
/// let mat = Matrix::<f32>::create(3, 3, vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])?;
/// let qr = qr_decomposition(mat.clone())?;
/// assert!(qr.0.conjugate_transpose()?.times(qr.1)?.equal(&mat));
/// # Ok(())
/// # }
/// ```
pub fn qr_decomposition<T>(mat: Matrix<T>) -> IResult<(Matrix<T>, Matrix<T>)>
where
    T: Fractional + std::cmp::PartialOrd,
{
    let mrow = mat.row();
    let mcol = mat.column();
    let mut q = Matrix::eyes(mrow, mrow)?;
    let mut r = mat.clone();
    let two = T::one() + T::one();

    for index in 1..=mcol.min(mrow) {
        // an is sub-column-vector for mat
        let mut an = r.get_col(index)?.map(&mut |e| e.clone())?;
        let ann = r.get_element(index, index)?.clone();
        // remove element over index
        for row in 1..=((index - 1).min(mrow)) {
            an.set_element(row, 1, T::zero())?;
        }
        let an_norm = euclid_norm(an.clone())?;
        // vn = an + sign(ann) ||an|| en
        let vn = an.clone().plus(
            identity_vector_column(mrow, index)?
                .muls(an_norm.clone())?
                // sign(x) = 1 if x >= 0 else -1
                .muls(if ann < T::zero() { -T::one() } else { T::one() })?,
        )?;
        // Hn = I - 2 (vn vn^H) / (vn^H v)
        let hn = Matrix::eyes(mrow, mrow)?.subtract(
            times_v(vn.clone(), vn.conjugate_transpose()?)?
                .muls(two.clone().ndiv(times_d(vn.conjugate_transpose()?, vn)?)?)?,
        )?;
        q = hn.clone().times(q)?;
        r = hn.times(r)?;
        // skip unnecessary calculation
        if is_upper_triangle_matrix(&r) {
            break;
        }
    }

    Ok((q, r))
}

/// qr decomposition
///
/// use Householder method and reduced
///
/// q^H * r = m
///
/// H is conjugate transpose
///
/// ```rust
/// # use rmatrix_ks::error::IResult;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::decompose::qr_decomposition_reduced;
/// # fn main() -> IResult<()> {
/// let mat = Matrix::<f32>::create(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// let qr = qr_decomposition_reduced(mat.clone())?;
/// assert_eq!(((2, 3), (2, 2)), (qr.0.dimensions(), qr.1.dimensions()));
/// assert!(qr.0.conjugate_transpose()?.times(qr.1)?.equal(&mat));
/// # Ok(())
/// # }
/// ```
pub fn qr_decomposition_reduced<T>(mat: Matrix<T>) -> IResult<(Matrix<T>, Matrix<T>)>
where
    T: Clone + Fractional + std::cmp::PartialOrd,
{
    let (q, r) = qr_decomposition(mat)?;
    let mut reduced_r = Matrix::zeros(r.rank()?, r.column())?;
    let mut reduced_q = Matrix::zeros(reduced_r.row(), q.column())?;
    // remove zero rows
    for row in 1..=reduced_r.row() {
        for col in 1..=reduced_r.column() {
            reduced_r.set_element(row, col, r.get_element(row, col)?.clone())?;
        }
        for col in 1..=reduced_q.column() {
            reduced_q.set_element(row, col, q.get_element(row, col)?.clone())?;
        }
    }
    Ok((reduced_q, reduced_r))
}

/// qr decomposition
///
/// use Gram-Schmidt method
///
/// q * r = m
///
/// ```rust
/// # use rmatrix_ks::error::IResult;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::decompose::qr_decomposition_gs;
/// # fn main() -> IResult<()> {
/// let mat = Matrix::<f32>::create(3, 3, vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])?;
/// let qr = qr_decomposition_gs(mat.clone())?;
/// assert!(qr.0.times(qr.1)?.equal(&mat));
/// # Ok(())
/// # }
/// ```
pub fn qr_decomposition_gs<T>(mat: Matrix<T>) -> IResult<(Matrix<T>, Matrix<T>)>
where
    T: Fractional,
{
    let mrow = mat.row();
    let mcol = mat.column();
    let mut an = Vec::with_capacity(mcol);
    for col in 1..=mcol {
        an.push(mat.get_col(col)?);
    }
    let mut un: Vec<ColumnVector<T>> = Vec::with_capacity(mcol);
    let mut en: Vec<ColumnVector<T>> = Vec::with_capacity(mcol);
    for col in 1..=mcol {
        // get a[i]
        let ai = match an.get(col - 1) {
            Some(element) => Ok(element),
            None => Err(IError::Message(format!(
                "read index {} out of boundary",
                col
            ))),
        }?
        .clone()
        .map(&mut |e| e.clone())?;

        // get u[i]
        let mut ui = ai.clone();
        for index in 1..=(col - 1) {
            let ek = match en.get(index - 1) {
                Some(element) => Ok(element),
                None => Err(IError::Message(format!(
                    "read index {} out of boundary",
                    index
                ))),
            }?
            .clone()
            .map(&mut |e| e.clone())?;

            ui = ui.subtract(
                ek.clone()
                    .muls(times_d(ai.clone().conjugate_transpose()?, ek)?)?,
            )?;
        }
        un.push(ui.clone());
        let norm = euclid_norm(ui.clone())?;
        let ei = ui.divs(norm)?;
        en.push(ei);
    }
    let mut q = Matrix::zeros(mrow, mcol)?;
    let mut r = Matrix::zeros(mcol, mcol)?;

    for row in 1..=mrow {
        for col in 1..=mcol {
            let ei = match en.get(col - 1) {
                Some(element) => Ok(element),
                None => Err(IError::Message(format!(
                    "read index {} out of boundary",
                    col
                ))),
            }?
            .clone();
            q.set_element(row, col, ei.get_element(row, 1)?.clone().clone())?;
        }
    }

    for col1 in 1..=mcol {
        let ei = match en.get(col1 - 1) {
            Some(element) => Ok(element),
            None => Err(IError::Message(format!(
                "read index {} out of boundary",
                col1
            ))),
        }?
        .clone()
        .map(&mut |e| e.clone())?;

        for col2 in col1..=mcol {
            let ai = match an.get(col2 - 1) {
                Some(element) => Ok(element),
                None => Err(IError::Message(format!(
                    "read index {} out of boundary",
                    col2
                ))),
            }?
            .clone()
            .map(&mut |e| e.clone())?;

            r.set_element(col1, col2, times_d(ai.conjugate_transpose()?, ei.clone())?)?;
        }
    }

    Ok((q, r))
}
