//! # Decompose
//!
//! some decomposition method

use crate::error::Error;
use crate::error::Result;
use crate::matrix::Matrix;
use crate::num::number::Fractional;
use crate::num::number::Number;
use crate::utils::predicate::is_lower_triangle_matrix;
use crate::utils::predicate::is_upper_triangle_matrix;
use crate::vector::euclid_norm;
use crate::vector::identity_vector_column;
use crate::vector::times_d;
use crate::vector::times_v;
use crate::vector::VectorC;

/// transform the square matrix to lower triangle form by rows elimination
pub(crate) fn lower_triangularize<T, const ROW: usize>(
    mat: Matrix<T, ROW, ROW>,
) -> Result<(Matrix<T, ROW, ROW>, Matrix<T, ROW, ROW>)>
where
    T: Number,
{
    let mut reduced = mat.to_owned();
    let mut p_all = Matrix::<T, ROW, ROW>::eyes()?;

    if !(is_lower_triangle_matrix(&reduced) || ROW < 2) {
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
/// # use rmatrix_ks::utils::decompose::plu_decomposition;
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
/// use Householder method
///
/// q^H * r = m
///
/// H is conjugate transpose
///
/// ```rust
/// # use rmatrix_ks::error::Result;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::decompose::qr_decomposition;
/// # fn main() -> Result<()> {
/// let mat = Matrix::<f32, 3, 3>::create(vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])?;
/// let qr = qr_decomposition(mat.to_owned())?;
/// assert!(qr.0.conjugate_transpose()?.times(qr.1)?.equal(&mat));
/// # Ok(())
/// # }
/// ```
pub fn qr_decomposition<T, const ROW: usize, const COL: usize>(
    mat: Matrix<T, ROW, COL>,
) -> Result<(Matrix<T, ROW, ROW>, Matrix<T, ROW, COL>)>
where
    T: Fractional + std::cmp::PartialOrd,
{
    let mut q = Matrix::<T, ROW, ROW>::eyes()?;
    let mut r = mat.to_owned();
    let two = T::one() + T::one();

    for index in 1..=COL.min(ROW) {
        // an is sub-column-vector for mat
        let mut an = r.get_col(index)?.map(&mut |e| e.to_owned())?;
        let ann = r.get_element(index, index)?.to_owned();
        // remove element over index
        for row in 1..=((index - 1).min(ROW)) {
            an.set_element(row, 1, T::zero())?;
        }
        let an_norm = euclid_norm(an.to_owned())?;
        // vn = an + sign(ann) ||an|| en
        let vn = an.to_owned().plus(
            identity_vector_column::<T, ROW>(index)?
                .muls(an_norm.to_owned())?
                // sign(x) = 1 if x >= 0 else -1
                .muls(if ann < T::zero() { -T::one() } else { T::one() })?,
        )?;
        // Hn = I - 2 (vn vn^H) / (vn^H v)
        let hn = Matrix::<T, ROW, ROW>::eyes()?.subtract(
            times_v(vn.to_owned(), vn.conjugate_transpose()?)?.muls(
                two.to_owned()
                    .ndiv(times_d(vn.conjugate_transpose()?, vn)?)?,
            )?,
        )?;
        q = hn.to_owned().times(q)?;
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
/// use Gram-Schmidt method
///
/// q * r = m
///
/// ```rust
/// # use rmatrix_ks::error::Result;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::decompose::qr_decomposition_gs;
/// # fn main() -> Result<()> {
/// let mat = Matrix::<f32, 3, 3>::create(vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])?;
/// let qr = qr_decomposition_gs(mat.to_owned())?;
/// assert!(qr.0.times(qr.1)?.equal(&mat));
/// # Ok(())
/// # }
/// ```
pub fn qr_decomposition_gs<T, const ROW: usize, const COL: usize>(
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
        .to_owned()
        .map(&mut |e| e.to_owned())?;

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
            .to_owned()
            .map(&mut |e| e.to_owned())?;

            ui = ui.subtract(
                ek.to_owned()
                    .muls(times_d(ai.to_owned().conjugate_transpose()?, ek)?)?,
            )?;
        }
        un.push(ui.to_owned());
        let norm = euclid_norm(ui.to_owned())?;
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
        .to_owned()
        .map(&mut |e| e.to_owned())?;

        for col2 in col1..=COL {
            let ai = match an.get(col2 - 1) {
                Some(element) => Ok(element),
                None => Err(Error::Message(format!(
                    "read index {} out of boundary",
                    col2
                ))),
            }?
            .to_owned()
            .map(&mut |e| e.to_owned())?;

            r.set_element(
                col1,
                col2,
                times_d(ai.conjugate_transpose()?, ei.to_owned())?,
            )?;
        }
    }

    Ok((q, r))
}
