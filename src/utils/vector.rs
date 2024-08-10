//! # Vector
//!
//! some vector manipulations

use crate::column_vector::ColumnVector;
use crate::error::Error;
use crate::error::Result;
use crate::matrix::Matrix;
use crate::num::number::Fractional;
use crate::num::number::Number;
use crate::row_vector::RowVector;

/// vector cross product
///
/// only 3-d vector has cross product
///
/// ```rust
/// # use rmatrix_ks::column_vector::ColumnVector;
/// # use rmatrix_ks::error::Result;
/// # use rmatrix_ks::utils::vector::times_c;
/// # fn main() -> Result<()> {
/// let vec1: ColumnVector<f32> = ColumnVector::create(3, vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: ColumnVector<f32> = ColumnVector::create(3, vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(ColumnVector::create(3, vec![-3.0f32, 6.0f32, -3.0f32])?, times_c(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn times_c<T>(vec1: ColumnVector<T>, vec2: ColumnVector<T>) -> Result<ColumnVector<T>>
where
    T: Number,
{
    if vec1.row != 3 {
        Err(Error::IncompatibleShape((3, 1), (vec1.row, 1)))
    } else if vec2.row != 3 {
        Err(Error::IncompatibleShape((3, 1), (vec2.row, 1)))
    } else {
        ColumnVector::create(
            3,
            vec![
                vec1.get_element(2)?.to_owned() * vec2.get_element(3)?.to_owned()
                    - vec1.get_element(3)?.to_owned() * vec2.get_element(2)?.to_owned(),
                vec1.get_element(3)?.to_owned() * vec2.get_element(1)?.to_owned()
                    - vec1.get_element(1)?.to_owned() * vec2.get_element(3)?.to_owned(),
                vec1.get_element(1)?.to_owned() * vec2.get_element(2)?.to_owned()
                    - vec1.get_element(2)?.to_owned() * vec2.get_element(1)?.to_owned(),
            ],
        )
    }
}

/// vector dot product
///
/// ```rust
/// # use rmatrix_ks::column_vector::ColumnVector;
/// # use rmatrix_ks::error::Result;
/// # use rmatrix_ks::row_vector::RowVector;
/// # use rmatrix_ks::utils::vector::times_d;
/// # fn main() -> Result<()> {
/// let vec1: RowVector<f32> = RowVector::create(3, vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: ColumnVector<f32> = ColumnVector::create(3, vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(32.0f32, times_d(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn times_d<T>(vec1: RowVector<T>, vec2: ColumnVector<T>) -> Result<T>
where
    T: Number,
{
    if vec1.col != vec2.row {
        Err(Error::IncompatibleShape((vec1.col, 1), (vec2.row, 1)))
    } else {
        Ok(vec1
            .inner
            .iter()
            .zip(vec2.inner.iter())
            .map(|(e1, e2)| e1.to_owned() * e2.to_owned())
            .sum())
    }
}

/// vector convolution
///
/// ```rust
/// # use rmatrix_ks::column_vector::ColumnVector;
/// # use rmatrix_ks::utils::vector::convolution;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let vec1: ColumnVector<f32> = ColumnVector::create(3, vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: ColumnVector<f32> = ColumnVector::create(3, vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(ColumnVector::create(5, vec![4.0f32, 13.0f32, 28.0f32, 27.0f32, 18.0f32])?,
///     convolution(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn convolution<T>(vec1: ColumnVector<T>, vec2: ColumnVector<T>) -> Result<ColumnVector<T>>
where
    T: Number,
{
    let edge: usize = vec1.row + vec2.row - 1;
    let mut conv: ColumnVector<T> = ColumnVector::zeros(edge)?;

    for k in 1..=edge {
        for i in 1..=(k.min(vec1.row)) {
            // k + 1 = i + j
            let j = k + 1 - i;
            if (1..=vec2.row).contains(&j) {
                conv.set_element(
                    k,
                    conv.get_element(k)?.to_owned()
                        + vec1.get_element(i)?.to_owned() * vec2.get_element(j)?.to_owned(),
                )?;
            }
        }
    }

    Ok(conv)
}

/// vector product
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::column_vector::ColumnVector;
/// # use rmatrix_ks::row_vector::RowVector;
/// # use rmatrix_ks::utils::vector::times_v;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let vec1: ColumnVector<f32> = ColumnVector::create(3, vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: RowVector<f32> = RowVector::create(3, vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(Matrix::create(3, 3,
///     vec![
///         4.0f32, 5.0f32, 6.0f32,
///         8.0f32, 10.0f32, 12.0f32,
///         12.0f32, 15.0f32, 18.0f32])?,
///     times_v(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn times_v<T>(vector_c: ColumnVector<T>, vector_r: RowVector<T>) -> Result<Matrix<T>>
where
    T: Number,
{
    let mut mat = Matrix::<T>::create(
        vector_c.row,
        vector_r.col,
        vec![T::default(); vector_c.row * vector_r.col],
    )?;
    for r in 1..=vector_c.row {
        for c in 1..=vector_r.col {
            mat.set_element(
                r,
                c,
                vector_c.get_element(r)?.to_owned() * vector_r.get_element(c)?.to_owned(),
            )?;
        }
    }
    Ok(mat)
}

/// euclid norm for vector
///
/// aka l2-norm
///
/// ```rust
/// # use rmatrix_ks::column_vector::ColumnVector;
/// # use rmatrix_ks::utils::vector::euclid_norm;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let v: ColumnVector<f32> = ColumnVector::create(2, vec![3.0f32, 4.0f32])?;
/// assert_eq!(5.0f32, euclid_norm(v)?);
/// # Ok(())
/// # }
/// ```
pub fn euclid_norm<T>(vc: ColumnVector<T>) -> Result<T>
where
    T: Fractional,
{
    times_d(vc.conjugate_transpose()?, vc)?.nsqrt()
}

/// root mean square for a vector
///
/// ```rust
/// # use rmatrix_ks::column_vector::ColumnVector;
/// # use rmatrix_ks::utils::vector::root_mean_square;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let v: ColumnVector<f32> = ColumnVector::create(2, vec![3.0f32, 4.0f32])?;
/// assert_eq!(5.0f32 / 2.0f32.sqrt(), root_mean_square(v)?);
/// # Ok(())
/// # }
/// ```
pub fn root_mean_square<T>(vc: ColumnVector<T>) -> Result<T>
where
    T: Fractional,
{
    let row = vc.row;
    euclid_norm(vc)?.ndiv(T::from_usize(row).nsqrt()?)
}
