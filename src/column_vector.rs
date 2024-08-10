//! # Column Vector
//!
//! basic row vector, Matrix(row, 1)

use crate::error::Error;
use crate::error::Result;
use crate::matrix::Matrix;
use crate::num::number::Equal;
use crate::num::number::Number;
use crate::num::number::One;
use crate::num::number::Zero;
use crate::row_vector::RowVector;

#[cfg(feature = "serde_mat")]
use serde::{Deserialize, Serialize};

/// column vector
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_mat", derive(Serialize, Deserialize))]
pub struct ColumnVector<T> {
    pub row: usize,
    pub(crate) inner: Vec<T>,
}

impl<T> ColumnVector<T> {
    /// create a column vector
    ///
    /// ```rust
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1}, {2}, {3}}
    /// let _ = ColumnVector::<i32>::create(3, vec![1, 2, 3])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create(row: usize, data: Vec<T>) -> Result<Self> {
        if row < data.len() {
            Err(Error::IncompatibleSizeError((row, 1), data.len()))
        } else {
            Ok(ColumnVector { row, inner: data })
        }
    }

    /// create a zero column vector with size row by 1
    ///
    /// ```rust
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{0}, {0}, {0}}
    /// let _ = ColumnVector::<i32>::zeros(3)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn zeros(row: usize) -> Result<Self>
    where
        T: Clone + Zero,
    {
        ColumnVector::create(row, vec![T::zero(); row])
    }

    /// get the value of the specific position of the column vector
    ///
    /// ```rust
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1i8}, {2i8}, {3i8}}
    /// let v: ColumnVector<i8> = ColumnVector::create(3, vec![1, 2, 3])?;
    /// assert_eq!(&2, v.get_element(2)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_element(&self, row: usize) -> Result<&T> {
        if row == 0 {
            Err(Error::OutOfBoundary(row, 1))
        } else {
            match self.inner.get(row - 1) {
                Some(element) => Ok(element),
                None => Err(Error::OutOfBoundary(row, 1)),
            }
        }
    }

    /// set the value of the specific position of the row vector
    ///
    /// ```rust
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{0i8}, {0i8}}
    /// let mut v: ColumnVector<i32> = ColumnVector::zeros(2)?;
    /// // {{3i32}, {0i32}}
    /// v.set_element(1, 3i32)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_element(&mut self, row: usize, replace: T) -> Result<()> {
        if row == 0 {
            Err(Error::OutOfBoundary(row, 1))
        } else {
            match self.inner.get_mut(row - 1) {
                Some(element) => {
                    *element = replace;
                    Ok(())
                }
                None => Err(Error::OutOfBoundary(row, 1)),
            }
        }
    }

    /// map a function to a column vector
    ///
    /// ```rust
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let mat: ColumnVector<i32> = ColumnVector::create(3, vec![1, 2, 3])?;
    /// let zero: ColumnVector<i32> = ColumnVector::create(3, vec![0i32; 3])?;
    /// assert_eq!(zero, mat.map(&mut |e| e * 0)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn map<N, F>(&self, f: &mut F) -> Result<ColumnVector<N>>
    where
        T: Clone,
        F: Fn(T) -> N,
    {
        ColumnVector::<N>::create(
            self.row,
            self.inner.iter().map(|e| f(e.to_owned())).collect(),
        )
    }

    /// transpose a column vector
    ///
    ///
    /// ```rust
    /// # use rmatrix_ks::error::Result;
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::row_vector::RowVector;
    /// # fn main() -> Result<()> {
    /// // {{1i32}, {2i32}, {3i32}, {4i32}, {5i32}, {6i32}}
    /// let v: ColumnVector<i32> = ColumnVector::create(6, vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!(RowVector::create(6, vec![1, 2, 3, 4, 5, 6])?,
    ///     v.transpose()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn transpose(&self) -> Result<RowVector<T>>
    where
        T: Clone,
    {
        let mut transposed = Vec::with_capacity(self.row);
        for r in 1..=self.row {
            transposed.push(self.get_element(r)?.to_owned());
        }
        RowVector::create(self.row, transposed)
    }

    /// conjugate transpose a column vector
    ///
    /// ```rust
    /// # use rmatrix_ks::cmplx;
    /// # use rmatrix_ks::error::Result;
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::row_vector::RowVector;
    /// # use rmatrix_ks::num::complex::Complex;
    /// # fn main() -> Result<()> {
    /// let v: ColumnVector<Complex<i32>> =
    ///     ColumnVector::create(2, vec![cmplx!(1, 2), cmplx!(2, 3)])?;
    /// assert_eq!(RowVector::create(2, vec![cmplx!(1, -2), cmplx!(2, -3)])?,
    ///     v.conjugate_transpose()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn conjugate_transpose(&self) -> Result<RowVector<T>>
    where
        T: Clone + Number,
    {
        let mut transposed = Vec::with_capacity(self.row);
        for r in 1..=self.row {
            transposed.push(self.get_element(r)?.to_owned().conjugate());
        }
        RowVector::create(self.row, transposed)
    }

    /// column vector addition
    ///
    /// ```rust
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let v1: ColumnVector<f32> = ColumnVector::create(6, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
    /// let v2: ColumnVector<f32> = ColumnVector::create(6, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
    /// assert_eq!(ColumnVector::create(6, vec![2.0f32, 4.0f32, 6.0f32, 8.0f32, 10.0f32, 12.0f32])?,
    ///     v1.plus(v2)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn plus(self, rhs: Self) -> Result<Self>
    where
        T: Number,
    {
        Self::create(
            self.row,
            self.inner
                .iter()
                .zip(rhs.inner.iter())
                .map(|(a, b)| a.to_owned() + b.to_owned())
                .collect::<Vec<_>>(),
        )
    }

    /// column vector addition with scalar
    ///
    /// ```rust
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let v: ColumnVector<f32> = ColumnVector::create(6, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
    /// assert_eq!(ColumnVector::create(6, vec![2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32])?,
    ///     v.adds(1.0f32)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn adds(self, scalar: T) -> Result<Self>
    where
        T: Number,
    {
        self.map(&mut |a| scalar.to_owned() + a)
    }

    /// column vector multiplication with scalar
    ///
    /// ```rust
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let v: ColumnVector<f32> = ColumnVector::create(6, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
    /// assert_eq!(ColumnVector::create(6, vec![2.0f32, 4.0f32, 6.0f32, 8.0f32, 10.0f32, 12.0f32])?,
    ///     v.muls(2.0f32)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn muls(self, scalar: T) -> Result<Self>
    where
        T: Number,
    {
        self.map(&mut |a| scalar.to_owned() * a)
    }

    /// column vector division with scalar
    ///
    /// ```rust
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let v: ColumnVector<f32> = ColumnVector::create(4, vec![2.0f32, 4.0f32, 6.0f32, 8.0f32])?;
    /// assert_eq!(ColumnVector::create(4, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?,
    ///     v.divs(2.0f32)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn divs(self, scalar: T) -> Result<Self>
    where
        T: Number,
    {
        let mut quotient = self.clone();
        for index in 0..self.row {
            match quotient.inner.get_mut(index) {
                Some(m) => match m.to_owned().ndiv(scalar.to_owned()) {
                    Ok(v) => {
                        *m = v;
                        Ok(())
                    }
                    Err(e) => Err(e),
                },
                None => Err(Error::Message(format!(
                    "read index {} out of boundary",
                    index
                ))),
            }?;
        }
        Ok(quotient)
    }

    /// column vector subtraction
    ///
    /// ```rust
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let a: ColumnVector<f32> = ColumnVector::create(4, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
    /// let b: ColumnVector<f32> = ColumnVector::create(4, vec![5.0f32, 6.0f32, 7.0f32, 8.0f32])?;
    /// assert_eq!(
    ///     ColumnVector::<f32>::create(4, vec![4.0f32, 4.0f32, 4.0f32, 4.0f32])?,
    ///     b.subtract(a)?
    /// );
    /// # Ok(())
    /// # }
    pub fn subtract(self, rhs: Self) -> Result<Self>
    where
        T: Number,
    {
        self.plus(rhs.map(&mut |e| -e)?)
    }
}

impl<T> TryFrom<Matrix<T>> for ColumnVector<T> {
    type Error = Error;
    fn try_from(value: Matrix<T>) -> Result<Self> {
        ColumnVector::create(value.row() * value.column(), value.inner)
    }
}

impl<T> std::cmp::PartialEq for ColumnVector<T>
where
    T: Equal,
{
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row
            && self
                .inner
                .iter()
                .take(self.row)
                .zip(other.inner.iter())
                .all(|(e1, e2)| e1.equal(e2))
    }
}

/// get the cloumn en
///
/// for column vector, e1(3) = {{1}, {0}, {0}},
/// and e2(4) = {{0}, {1}, {0}, {0}}
///
/// ```rust
/// # use rmatrix_ks::column_vector::ColumnVector;
/// # use rmatrix_ks::column_vector::identity_vector_column;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// assert_eq!(ColumnVector::create(3, vec![1.0, 0.0, 0.0])?,
///     identity_vector_column::<f32>(3, 1)?);
/// assert_eq!(ColumnVector::create(4, vec![0.0, 1.0, 0.0, 0.0])?,
///     identity_vector_column::<f32>(4, 2)?);
/// # Ok(())
/// # }
/// ```
pub fn identity_vector_column<T>(row: usize, index: usize) -> Result<ColumnVector<T>>
where
    T: Clone + Zero + One,
{
    let mut vector = ColumnVector::zeros(row)?;
    vector.set_element(index, T::one())?;
    Ok(vector)
}
