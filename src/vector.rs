//! # Vector
//!
//! some vector manipulations
//!
//! ColumnVector<T, row> = Matrix<T, row, 1>
//!
//! RowVector<T, col> = Matrix<T, 1, col>

use crate::error::IError;
use crate::error::IResult;
use crate::matrix::Matrix;

/// type alias for row vector
pub type RowVector<T> = Matrix<T>;

/// type alias for column vector
pub type ColumnVector<T> = Matrix<T>;

/// get the row en
///
/// for row vector, e1(3) = {{1, 0, 0}},
/// and e2(4) = {{0, 1, 0, 0}}
///
/// ```rust
/// # use rmatrix_ks::error::IResult;
/// # use rmatrix_ks::vector::identity_vector_row;
/// # use rmatrix_ks::vector::RowVector;
/// # fn main() -> IResult<()> {
/// assert_eq!(RowVector::create(1, 3, vec![1.0, 0.0, 0.0])?,
///     identity_vector_row::<f32>(3, 1)?);
/// assert_eq!(RowVector::create(1, 4, vec![0.0, 1.0, 0.0, 0.0])?,
///     identity_vector_row::<f32>(4, 2)?);
/// # Ok(())
/// # }
/// ```
pub fn identity_vector_row<T>(column_size: usize, index: usize) -> IResult<RowVector<T>>
where
    T: std::clone::Clone + std::default::Default + crate::num::number::IOne,
{
    let mut vector = RowVector::defaults(1, column_size)?;
    vector.set_element(1, index, T::one())?;
    Ok(vector)
}

/// get the cloumn en
///
/// for column vector, e1(3) = {{1}, {0}, {0}},
/// and e2(4) = {{0}, {1}, {0}, {0}}
///
/// ```rust
/// # use rmatrix_ks::error::IResult;
/// # use rmatrix_ks::vector::ColumnVector;
/// # use rmatrix_ks::vector::identity_vector_column;
/// # fn main() -> IResult<()> {
/// assert_eq!(ColumnVector::create(3, 1, vec![1.0, 0.0, 0.0])?,
///     identity_vector_column::<f32>(3, 1)?);
/// assert_eq!(ColumnVector::create(4, 1, vec![0.0, 1.0, 0.0, 0.0])?,
///     identity_vector_column::<f32>(4, 2)?);
/// # Ok(())
/// # }
/// ```
pub fn identity_vector_column<T>(row_size: usize, index: usize) -> IResult<ColumnVector<T>>
where
    T: std::clone::Clone + std::default::Default + crate::num::number::IOne,
{
    let mut vector = ColumnVector::defaults(row_size, 1)?;
    vector.set_element(index, 1, T::one())?;
    Ok(vector)
}

/// vector cross product
///
/// only 3-d vector has cross product
///
/// ```rust
/// # use rmatrix_ks::error::IResult;
/// # use rmatrix_ks::vector::ColumnVector;
/// # use rmatrix_ks::vector::times_c;
/// # fn main() -> IResult<()> {
/// let vec1: ColumnVector<f32> = ColumnVector::create(3, 1, vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: ColumnVector<f32> = ColumnVector::create(3, 1, vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(ColumnVector::create(3, 1, vec![-3.0f32, 6.0f32, -3.0f32])?, times_c(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn times_c<T>(vector_a: ColumnVector<T>, vector_b: ColumnVector<T>) -> IResult<ColumnVector<T>>
where
    T: crate::num::number::INumber,
{
    if vector_a.row() != 3 {
        Err(IError::IncompatibleShape((3, 1), (vector_a.row(), 1)))
    } else if vector_b.row() != 3 {
        Err(IError::IncompatibleShape((3, 1), (vector_b.row(), 1)))
    } else {
        ColumnVector::create(
            3,
            1,
            vec![
                vector_a.get_element(2, 1)?.clone() * vector_b.get_element(3, 1)?.clone()
                    - vector_a.get_element(3, 1)?.clone() * vector_b.get_element(2, 1)?.clone(),
                vector_a.get_element(3, 1)?.clone() * vector_b.get_element(1, 1)?.clone()
                    - vector_a.get_element(1, 1)?.clone() * vector_b.get_element(3, 1)?.clone(),
                vector_a.get_element(1, 1)?.clone() * vector_b.get_element(2, 1)?.clone()
                    - vector_a.get_element(2, 1)?.clone() * vector_b.get_element(1, 1)?.clone(),
            ],
        )
    }
}

/// vector dot product
///
/// ```rust
/// # use rmatrix_ks::error::IResult;
/// # use rmatrix_ks::vector::ColumnVector;
/// # use rmatrix_ks::vector::RowVector;
/// # use rmatrix_ks::vector::times_d;
/// # fn main() -> IResult<()> {
/// let vec1: RowVector<f32> = RowVector::create(1, 3, vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: ColumnVector<f32> = ColumnVector::create(3, 1, vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(32.0f32, times_d(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn times_d<T>(vector_a: RowVector<T>, vector_b: ColumnVector<T>) -> IResult<T>
where
    T: crate::num::number::INumber,
{
    if vector_a.column() != vector_b.row() {
        Err(IError::IncompatibleShape(
            (vector_a.column(), 1),
            (vector_b.row(), 1),
        ))
    } else {
        Ok(vector_a
            .get_inner()
            .zip(vector_b.get_inner())
            .map(|(e1, e2)| e1.clone() * e2.clone())
            .sum())
    }
}

/// vector convolution
///
/// ```rust
/// # use rmatrix_ks::vector::ColumnVector;
/// # use rmatrix_ks::vector::convolution;
/// # use rmatrix_ks::error::IResult;
/// # fn main() -> IResult<()> {
/// let vec1: ColumnVector<f32> = ColumnVector::create(3, 1, vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: ColumnVector<f32> = ColumnVector::create(3, 1, vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(ColumnVector::create(5, 1, vec![4.0f32, 13.0f32, 28.0f32, 27.0f32, 18.0f32])?,
///     convolution(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn convolution<T>(
    vector_a: ColumnVector<T>,
    vector_b: ColumnVector<T>,
) -> IResult<ColumnVector<T>>
where
    T: crate::num::number::INumber,
{
    let edge: usize = vector_a.row() + vector_b.row() - 1;
    let mut conv: ColumnVector<T> = ColumnVector::defaults(edge, 1)?;

    for k in 1..=edge {
        for i in 1..=(k.min(vector_a.row())) {
            // k + 1 = i + j
            let j = k + 1 - i;
            if (1..=vector_b.row()).contains(&j) {
                conv.set_element(
                    k,
                    1,
                    conv.get_element(k, 1)?.clone()
                        + vector_a.get_element(i, 1)?.clone() * vector_b.get_element(j, 1)?.clone(),
                )?;
            }
        }
    }

    Ok(conv)
}

/// vector product
///
/// ```rust
/// # use rmatrix_ks::error::IResult;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::vector::ColumnVector;
/// # use rmatrix_ks::vector::RowVector;
/// # use rmatrix_ks::vector::times_v;
/// # fn main() -> IResult<()> {
/// let vec1: ColumnVector<f32> = ColumnVector::create(3, 1, vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: RowVector<f32> = RowVector::create(1, 3, vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(Matrix::create(3, 3,
///     vec![
///         4.0f32, 5.0f32, 6.0f32,
///         8.0f32, 10.0f32, 12.0f32,
///         12.0f32, 15.0f32, 18.0f32])?,
///     times_v(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn times_v<T>(vector_c: ColumnVector<T>, vector_r: RowVector<T>) -> IResult<Matrix<T>>
where
    T: crate::num::number::INumber,
{
    let mut mat = Matrix::<T>::create(
        vector_c.row(),
        vector_r.column(),
        vec![T::default(); vector_c.row() * vector_r.column()],
    )?;
    for r in 1..=vector_c.row() {
        for c in 1..=vector_r.column() {
            mat.set_element(
                r,
                c,
                vector_c.get_element(r, 1)?.clone() * vector_r.get_element(1, c)?.clone(),
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
/// # use rmatrix_ks::error::IResult;
/// # use rmatrix_ks::vector::ColumnVector;
/// # use rmatrix_ks::vector::euclid_norm;
/// # fn main() -> IResult<()> {
/// let v: ColumnVector<f32> = ColumnVector::create(2, 1, vec![3.0f32, 4.0f32])?;
/// assert_eq!(5.0f32, euclid_norm(v)?);
/// # Ok(())
/// # }
/// ```
pub fn euclid_norm<T>(column_vector: ColumnVector<T>) -> IResult<T>
where
    T: crate::num::number::IFractional,
{
    times_d(column_vector.conjugate_transpose()?, column_vector)?.nsqrt()
}

/// root mean square for a vector
///
/// ```rust
/// # use rmatrix_ks::error::IResult;
/// # use rmatrix_ks::vector::ColumnVector;
/// # use rmatrix_ks::vector::root_mean_square;
/// # fn main() -> IResult<()> {
/// let v: ColumnVector<f32> = ColumnVector::create(2, 1, vec![3.0f32, 4.0f32])?;
/// assert_eq!(5.0f32 / 2.0f32.sqrt(), root_mean_square(v)?);
/// # Ok(())
/// # }
/// ```
pub fn root_mean_square<T>(column_vector: ColumnVector<T>) -> IResult<T>
where
    T: crate::num::number::IFractional,
{
    let row = column_vector.row();
    euclid_norm(column_vector)?.ndiv(T::from_usize(row).nsqrt()?)
}
