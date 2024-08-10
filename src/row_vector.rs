//! # Row Vector
//!
//! basic row vector, Matrix(1, col)

use crate::column_vector::ColumnVector;
use crate::error::Error;
use crate::error::Result;
use crate::matrix::Matrix;
use crate::num::number::Equal;
use crate::num::number::Number;
use crate::num::number::One;
use crate::num::number::Zero;

#[cfg(feature = "serde_mat")]
use serde::{Deserialize, Serialize};

/// row vector
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_mat", derive(Serialize, Deserialize))]
pub struct RowVector<T> {
    pub col: usize,
    pub(crate) inner: Vec<T>,
}

impl<T> RowVector<T> {
    /// create a column vector
    ///
    /// ```rust
    /// # use rmatrix_ks::row_vector::RowVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1, 2, 3}}
    /// let _ = RowVector::<i32>::create(3, vec![1, 2, 3])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create(col: usize, data: Vec<T>) -> Result<Self> {
        if col < data.len() {
            Err(Error::IncompatibleSizeError((1, col), data.len()))
        } else {
            Ok(RowVector { col, inner: data })
        }
    }

    /// create a zero row vector with size 1 by col
    ///
    /// ```rust
    /// # use rmatrix_ks::row_vector::RowVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{0, 0, 0}}
    /// let _ = RowVector::<i32>::zeros(3)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn zeros(col: usize) -> Result<Self>
    where
        T: Clone + Zero,
    {
        RowVector::create(col, vec![T::zero(); col])
    }

    /// get the value of the specific position of the row vector
    ///
    /// ```rust
    /// # use rmatrix_ks::row_vector::RowVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1i8, 2i8, 3i8}}
    /// let v: RowVector<i8> = RowVector::create(3, vec![1, 2, 3])?;
    /// assert_eq!(&2, v.get_element(2)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_element(&self, col: usize) -> Result<&T> {
        if col == 0 {
            Err(Error::OutOfBoundary(1, col))
        } else {
            match self.inner.get(col - 1) {
                Some(element) => Ok(element),
                None => Err(Error::OutOfBoundary(1, col)),
            }
        }
    }

    /// set the value of the specific position of the row vector
    ///
    /// ```rust
    /// # use rmatrix_ks::row_vector::RowVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{0i32, 0i32}}
    /// let mut v: RowVector<i32> = RowVector::zeros(2)?;
    /// // {{3i32, 0i32}}
    /// v.set_element(1, 3i32)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_element(&mut self, col: usize, replace: T) -> Result<()> {
        if col == 0 {
            Err(Error::OutOfBoundary(col, 1))
        } else {
            match self.inner.get_mut(col - 1) {
                Some(element) => {
                    *element = replace;
                    Ok(())
                }
                None => Err(Error::OutOfBoundary(1, col)),
            }
        }
    }

    /// map a function to a row vector
    ///
    /// ```rust
    /// # use rmatrix_ks::row_vector::RowVector;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let mat: RowVector<i32> = RowVector::create(3, vec![1, 2, 3])?;
    /// let zero: RowVector<i32> = RowVector::create(3, vec![0i32; 3])?;
    /// assert_eq!(zero, mat.map(&mut |e| e * 0)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn map<N, F>(&self, f: &mut F) -> Result<RowVector<N>>
    where
        T: Clone,
        F: Fn(T) -> N,
    {
        RowVector::<N>::create(
            self.col,
            self.inner.iter().map(|e| f(e.to_owned())).collect(),
        )
    }

    /// transpose a row vector
    ///
    ///
    /// ```rust
    /// # use rmatrix_ks::error::Result;
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::row_vector::RowVector;
    /// # fn main() -> Result<()> {
    /// let v: RowVector<i32> = RowVector::create(6, vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!(ColumnVector::create(6, vec![1, 2, 3, 4, 5, 6])?,
    ///     v.transpose()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn transpose(&self) -> Result<ColumnVector<T>>
    where
        T: Clone,
    {
        let mut transposed = Vec::with_capacity(self.col);
        for c in 1..=self.col {
            transposed.push(self.get_element(c)?.to_owned());
        }
        ColumnVector::create(self.col, transposed)
    }

    /// conjugate transpose a row vector
    ///
    /// ```rust
    /// # use rmatrix_ks::cmplx;
    /// # use rmatrix_ks::error::Result;
    /// # use rmatrix_ks::column_vector::ColumnVector;
    /// # use rmatrix_ks::row_vector::RowVector;
    /// # use rmatrix_ks::num::complex::Complex;
    /// # fn main() -> Result<()> {
    /// let v: RowVector<Complex<i32>> =
    ///     RowVector::create(2, vec![cmplx!(1, 2), cmplx!(2, 3)])?;
    /// assert_eq!(ColumnVector::create(2, vec![cmplx!(1, -2), cmplx!(2, -3)])?,
    ///     v.conjugate_transpose()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn conjugate_transpose(&self) -> Result<ColumnVector<T>>
    where
        T: Clone + Number,
    {
        let mut transposed = Vec::with_capacity(self.col);
        for c in 1..=self.col {
            transposed.push(self.get_element(c)?.to_owned().conjugate());
        }
        ColumnVector::create(self.col, transposed)
    }

    /// row vector addition
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let mat1: Matrix<f32> = Matrix::create(2, 3, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
    /// let mat2: Matrix<f32> = Matrix::create(2, 3, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
    /// assert_eq!(Matrix::create(2, 3, vec![2.0f32, 4.0f32, 6.0f32, 8.0f32, 10.0f32, 12.0f32])?,
    ///     mat1.plus(mat2)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn plus(self, rhs: Self) -> Result<Self>
    where
        T: Number,
    {
        Self::create(
            self.col,
            self.inner
                .iter()
                .zip(rhs.inner.iter())
                .map(|(a, b)| a.to_owned() + b.to_owned())
                .collect::<Vec<_>>(),
        )
    }

    /// row vector addition with scalar
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let mat: Matrix<f32> = Matrix::create(2, 3, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
    /// assert_eq!(Matrix::create(2, 3, vec![2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32])?,
    ///     mat.adds(1.0f32)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn adds(self, scalar: T) -> Result<Self>
    where
        T: Number,
    {
        self.map(&mut |a| scalar.to_owned() + a)
    }

    /// row vector multiplication with scalar
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let mat: Matrix<f32> = Matrix::create(2, 3, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
    /// assert_eq!(Matrix::create(2, 3, vec![2.0f32, 4.0f32, 6.0f32, 8.0f32, 10.0f32, 12.0f32])?,
    ///     mat.muls(2.0f32)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn muls(self, scalar: T) -> Result<Self>
    where
        T: Number,
    {
        self.map(&mut |a| scalar.to_owned() * a)
    }

    /// row vector division with scalar
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let mat: Matrix<f32> = Matrix::create(2, 2, vec![2.0f32, 4.0f32, 6.0f32, 8.0f32])?;
    /// assert_eq!(Matrix::create(2, 2, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?,
    ///     mat.divs(2.0f32)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn divs(self, scalar: T) -> Result<Self>
    where
        T: Number,
    {
        let mut quotient = self.clone();
        for index in 0..self.col {
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

    /// row vector subtraction
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let a: Matrix<f32> = Matrix::create(2, 2, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
    /// let b: Matrix<f32> = Matrix::create(2, 2, vec![5.0f32, 6.0f32, 7.0f32, 8.0f32])?;
    /// assert_eq!(
    ///     Matrix::<f32>::create(2, 2, vec![4.0f32, 4.0f32, 4.0f32, 4.0f32])?,
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

impl<T> From<Matrix<T>> for RowVector<T> {
    fn from(value: Matrix<T>) -> Self {
        RowVector {
            col: value.row() * value.column(),
            inner: value.inner,
        }
    }
}

impl<T> std::cmp::PartialEq for RowVector<T>
where
    T: Equal,
{
    fn eq(&self, other: &Self) -> bool {
        self.col == other.col
            && self
                .inner
                .iter()
                .take(self.col)
                .zip(other.inner.iter())
                .all(|(e1, e2)| e1.equal(e2))
    }
}

/// get the row en
///
/// for row vector, e1(3) = {{1, 0, 0}},
/// and e2(4) = {{0, 1, 0, 0}}
///
/// ```rust
/// # use rmatrix_ks::row_vector::RowVector;
/// # use rmatrix_ks::row_vector::identity_vector_row;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// assert_eq!(RowVector::create(3, vec![1.0, 0.0, 0.0])?,
///     identity_vector_row::<f32>(3, 1)?);
/// assert_eq!(RowVector::create(4, vec![0.0, 1.0, 0.0, 0.0])?,
///     identity_vector_row::<f32>(4, 2)?);
/// # Ok(())
/// # }
/// ```
pub fn identity_vector_row<T>(col: usize, index: usize) -> Result<RowVector<T>>
where
    T: Clone + Zero + One,
{
    let mut vector = RowVector::zeros(col)?;
    vector.set_element(index, T::one())?;
    Ok(vector)
}
