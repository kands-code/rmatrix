//! # Matrix
//!
//! basic matrix with some common manipulations and features
//!
//! *all indices will start from 1*
//!
//! *do not have any optimization*

use crate::column_vector::ColumnVector;
use crate::error::Error;
use crate::error::Result;
use crate::num::number::Equal;
use crate::num::number::Number;
use crate::num::number::Zero;
use crate::row_vector::RowVector;
use crate::utils::predicate::is_upper_triangle_matrix;
use crate::utils::vector::times_v;

#[cfg(feature = "rand_mat")]
use rand::{
    distributions::{
        uniform::{SampleRange, SampleUniform},
        Distribution, Standard,
    },
    Rng,
};

#[cfg(feature = "serde_mat")]
use serde::{Deserialize, Serialize};

/// matrix type
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde_mat", derive(Serialize, Deserialize))]
pub struct Matrix<T> {
    /// inner data
    pub(crate) dim: (usize, usize),
    pub(crate) inner: Vec<T>,
}

/// default implementations
impl<T> Matrix<T> {
    /// create a matrix with size row by col
    ///
    /// **note: this will consume the original vector**
    ///
    /// this is the default constructor of matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1i8, 2i8}, {3i8, 4i8}}
    /// let _: Matrix<i8> = Matrix::create(2, 2, vec![1, 2, 3, 4])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create(row: usize, col: usize, data: Vec<T>) -> Result<Self> {
        if data.len() < row * col {
            Err(Error::IncompatibleSizeError((row, col), data.len()))
        } else {
            Ok(Matrix::<T> {
                dim: (row, col),
                inner: data,
            })
        }
    }

    /// create a zero matrix with size row by col
    ///
    /// this constructor is similar to Default, but the return value is Result
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{0i32, 0i32}, {0i32, 0i32}}
    /// let _: Matrix<i32> = Matrix::zeros(2, 2)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn zeros(row: usize, col: usize) -> Result<Self>
    where
        T: Clone + Zero,
    {
        Self::create(row, col, vec![T::zero(); row * col])
    }

    /// get the row size of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
    /// let mat: Matrix<i8> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!(2, mat.row());
    /// # Ok(())
    /// # }
    /// ```
    pub const fn row(&self) -> usize {
        self.dim.0
    }

    /// get the column size of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
    /// let mat: Matrix<i8> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!(3, mat.column());
    /// # Ok(())
    /// # }
    /// ```
    pub const fn column(&self) -> usize {
        self.dim.1
    }

    /// get the shape of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
    /// let mat: Matrix<i8> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!((2, 3), mat.dimensions());
    /// # Ok(())
    /// # }
    /// ```
    pub const fn dimensions(&self) -> (usize, usize) {
        self.dim
    }

    /// equlity check for matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::error::Result;
    /// use rmatrix_ks::num::number::Equal;
    /// # use rmatrix_ks::matrix::Matrix;
    /// # fn main() -> Result<()> {
    /// let m1 = Matrix::<f32>::create(2, 2, vec![1.2f32, 0.9f32, 0.7f32, 0.5f32])?;
    /// let m2 = Matrix::<f32>::create(2, 2, vec![0.9f32, 0.6f32, 0.4f32, 0.2f32])?;
    /// assert!(m1.equal(&m2.adds(0.3f32)?));
    /// # Ok(())
    /// # }
    /// ```
    pub fn equal(&self, rhs: &Self) -> bool
    where
        T: Equal,
    {
        if self.dim != rhs.dim {
            false
        } else {
            self.inner
                .iter()
                .take(self.row() * rhs.column())
                .zip(rhs.inner.iter())
                .all(|(e1, e2)| e1.equal(e2))
        }
    }

    /// convert index into inner index
    pub(crate) const fn to_inner_index(mcol: usize, row: usize, col: usize) -> usize {
        (row - 1) * mcol + col - 1
    }

    /// get the value of the specific position of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
    /// let mat: Matrix<i8> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!(&5, mat.get_element(2, 2)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_element(&self, row: usize, col: usize) -> Result<&T> {
        if row == 0 || col == 0 {
            Err(Error::OutOfBoundary(row, col))
        } else {
            match self
                .inner
                .get(Self::to_inner_index(self.column(), row, col))
            {
                Some(element) => Ok(element),
                None => Err(Error::OutOfBoundary(row, col)),
            }
        }
    }

    /// set the value of the specific position of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{0i32, 0i32}, {0i32, 0i32}}
    /// let mut mat: Matrix<i32> = Matrix::zeros(2, 2)?;
    /// // {{3i32, 0i32}, {0i32, 0i32}}
    /// mat.set_element(1, 1, 3i32)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_element(&mut self, row: usize, col: usize, replace: T) -> Result<()> {
        if row == 0 || col == 0 {
            Err(Error::OutOfBoundary(row, col))
        } else {
            let mcol = self.column();
            match self.inner.get_mut(Self::to_inner_index(mcol, row, col)) {
                Some(element) => {
                    *element = replace;
                    Ok(())
                }
                None => Err(Error::OutOfBoundary(row, col)),
            }
        }
    }

    /// get the n-th row of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
    /// let mat: Matrix<i8> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
    /// // {{4i8, 5i8, 6i8}}
    /// let _ = mat.get_row(2)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_row(&self, row: usize) -> Result<RowVector<&T>> {
        if row > self.row() {
            Err(Error::OutOfBoundary(row, 1))
        } else {
            let mut nth_row = Vec::with_capacity(self.column());
            for c in 1..=self.column() {
                nth_row.push(self.get_element(row, c)?);
            }
            RowVector::<&T>::create(self.column(), nth_row)
        }
    }

    /// get the n-th col of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
    /// let mat: Matrix<i8> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
    /// // {{2i8, 5i8}}
    /// let _ = mat.get_col(2)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_col(&self, col: usize) -> Result<ColumnVector<&T>> {
        if col > self.column() {
            Err(Error::OutOfBoundary(1, col))
        } else {
            let mut nth_col = Vec::with_capacity(self.column());
            for r in 1..=self.row() {
                nth_col.push(self.get_element(r, col)?);
            }
            ColumnVector::<&T>::create(self.row(), nth_col)
        }
    }

    /// create a diagonal matrix with size row by col
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1i32, 0i32}, {0i32, 2i32}}
    /// let _: Matrix<i32> = Matrix::diag(2, 2, vec![1, 2])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn diag(row: usize, col: usize, data: Vec<T>) -> Result<Self>
    where
        T: Clone + Zero,
    {
        if data.len() < row.min(col) {
            Err(Error::IncompatibleSizeError((row, col), data.len()))
        } else {
            let mut diag_mat = Self::zeros(row, col)?;
            for index in 1..=row.min(col) {
                match data.get(index - 1) {
                    Some(v) => diag_mat.set_element(index, index, v.to_owned()),
                    None => Err(Error::IncompatibleSizeError((row, col), data.len())),
                }?;
            }
            Ok(diag_mat)
        }
    }

    /// get the main diagonal of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
    /// let mat: Matrix<i8> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
    /// // {{1i8, 5i8}}
    /// let _ = mat.get_diag()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_diag(&self) -> Result<ColumnVector<&T>> {
        let edge: usize = self.row().min(self.column());
        let mut diag = Vec::with_capacity(edge);
        for i in 1..=edge {
            diag.push(self.get_element(i, i)?);
        }
        ColumnVector::<&T>::create(edge, diag)
    }

    /// transpose a matrix
    ///
    /// **note: this will consume the original matrix**
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
    /// let mat: Matrix<i8> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
    /// // {{1i8, 4i8}, {2i8, 5i8}, {3i8, 6i8}}
    /// assert_eq!((3, 2), mat.transpose()?.dimensions());
    /// # Ok(())
    /// # }
    /// ```
    pub fn transpose(&self) -> Result<Matrix<T>>
    where
        T: Clone,
    {
        let mut transposed = Vec::with_capacity(self.row() * self.column());
        for c in 1..=self.column() {
            for r in 1..=self.row() {
                transposed.push(self.get_element(r, c)?.to_owned());
            }
        }
        Matrix::<T>::create(self.column(), self.row(), transposed)
    }

    /// map a function to a matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let mat: Matrix<i32> = Matrix::create(2, 3, vec![1, 2, 3, 4, 5, 6])?;
    /// let zero: Matrix<i32> = Matrix::create(2, 3, vec![0i32; 6])?;
    /// assert_eq!(zero, mat.map(&mut |e| e * 0)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn map<N, F>(&self, f: &mut F) -> Result<Matrix<N>>
    where
        T: Clone,
        F: Fn(T) -> N,
    {
        Matrix::<N>::create(
            self.row(),
            self.column(),
            self.inner.iter().map(|e| f(e.to_owned())).collect(),
        )
    }
}

/// implementation for matrix which element type is a Number
impl<T> Matrix<T>
where
    T: Number,
{
    /// create an identity matrix with size row by col
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // {{1.0f32, 0.0f32}, {0.0f32, 1.0f32}}
    /// let _: Matrix<f32> = Matrix::eyes(2, 2)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn eyes(row: usize, col: usize) -> Result<Self> {
        let mut mat = Self::zeros(row, col)?;
        for i in 1..=row.min(col) {
            mat.set_element(i, i, T::one())?;
        }
        Ok(mat)
    }

    #[cfg(feature = "rand_mat")]
    /// create a random matrix with size row by col
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// // need "rand_mat" feature
    /// let _: Matrix<f32> = Matrix::rand(2, 2, -1.0..1.0)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn rand<R>(row: usize, col: usize, range: R) -> Result<Self>
    where
        R: std::ops::RangeBounds<T> + SampleRange<T> + Clone,
        T: Number + std::cmp::PartialOrd + SampleUniform,
        Standard: Distribution<T>,
    {
        Self::create(
            row,
            col,
            (1..=row * col)
                .map(|_| rand::thread_rng().gen_range(range.to_owned()))
                .collect(),
        )
    }

    /// exchange i row with j row
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let m = Matrix::<f32>::create(2, 2, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
    /// let p = Matrix::<f32>::p_change(2, 1, 2)?;
    /// assert_eq!(
    ///     Matrix::<f32>::create(2, 2, vec![3.0f32, 4.0f32, 1.0f32, 2.0f32])?,
    ///     p.to_owned().times(m.to_owned())?
    /// );
    /// assert_eq!(
    ///     Matrix::<f32>::create(2, 2, vec![2.0f32, 1.0f32, 4.0f32, 3.0f32])?,
    ///     m.times(p)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn p_change(row: usize, i: usize, j: usize) -> Result<Matrix<T>> {
        let mut mat = Matrix::<T>::eyes(row, row)?;
        mat.set_element(i, i, T::zero())?;
        mat.set_element(j, j, T::zero())?;
        mat.set_element(i, j, T::one())?;
        mat.set_element(j, i, T::one())?;
        Ok(mat)
    }

    /// multiply the i row of the matrix by a scalar
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let m = Matrix::<f32>::create(2, 2, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
    /// let p = Matrix::<f32>::p_muls(2, 1, 2.0f32)?;
    /// assert_eq!(
    ///     Matrix::<f32>::create(2, 2, vec![2.0f32, 4.0f32, 3.0f32, 4.0f32])?,
    ///     p.to_owned().times(m.to_owned())?
    /// );
    /// assert_eq!(
    ///     Matrix::<f32>::create(2, 2, vec![2.0f32, 2.0f32, 6.0f32, 4.0f32])?,
    ///     m.times(p)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn p_muls(row: usize, i: usize, k: T) -> Result<Matrix<T>> {
        let mut mat = Matrix::<T>::eyes(row, row)?;
        mat.set_element(i, i, k)?;
        Ok(mat)
    }

    /// add k times of the i row to the j row
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let m = Matrix::<f32>::create(2, 2, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
    /// let p = Matrix::<f32>::p_add(2, 1, 2, 1.0f32)?;
    /// assert_eq!(
    ///     Matrix::<f32>::create(2, 2, vec![1.0f32, 2.0f32, 4.0f32, 6.0f32])?,
    ///     p.to_owned().times(m.to_owned())?
    /// );
    /// assert_eq!(
    ///     Matrix::<f32>::create(2, 2, vec![1.0f32, 3.0f32, 3.0f32, 7.0f32])?,
    ///     m.times(p.transpose()?)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn p_add(row: usize, i: usize, j: usize, k: T) -> Result<Matrix<T>> {
        let mut mat = Matrix::<T>::eyes(row, row)?;
        mat.set_element(j, i, k)?;
        Ok(mat)
    }

    /// matrix addition
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
    pub fn plus(self, rhs: Self) -> Result<Self> {
        Self::create(
            self.row(),
            self.column(),
            self.inner
                .iter()
                .zip(rhs.inner.iter())
                .map(|(a, b)| a.to_owned() + b.to_owned())
                .collect::<Vec<_>>(),
        )
    }

    /// matrix addition with scalar
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
    pub fn adds(self, scalar: T) -> Result<Self> {
        self.map(&mut |a| scalar.to_owned() + a)
    }

    /// matrix multiplication
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let a: Matrix<f32> = Matrix::create(2, 2, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
    /// let b: Matrix<f32> = Matrix::create(2, 2, vec![5.0f32, 6.0f32, 7.0f32, 8.0f32])?;
    /// assert_eq!(
    ///     Matrix::<f32>::create(2, 2, vec![19.0f32, 22.0f32, 43.0f32, 50.0f32])?,
    ///     a.times(b)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn times(self, rhs: Matrix<T>) -> Result<Matrix<T>> {
        if self.column() != rhs.row() {
            Err(Error::IncompatibleShape(
                (self.column(), rhs.column()),
                (rhs.row(), rhs.column()),
            ))
        } else {
            let mut product = Matrix::<T>::create(
                self.row(),
                rhs.column(),
                vec![T::zero(); self.row() * rhs.column()],
            )?;
            for c in 1..=self.column() {
                product = product
                    .plus(times_v(
                        self.get_col(c)?.map(&mut |e: &T| e.to_owned())?,
                        rhs.get_row(c)?.map(&mut |e| e.to_owned())?,
                    )?)?
                    .to_owned();
            }
            Ok(product)
        }
    }

    /// matrix multiplication with scalar
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
    pub fn muls(self, scalar: T) -> Result<Self> {
        self.map(&mut |a| scalar.to_owned() * a)
    }

    /// matrix division with scalar
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
    pub fn divs(self, scalar: T) -> Result<Self> {
        let mut quotient = self.clone();
        for index in 0..(self.row() * self.column()) {
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

    /// matrix subtraction
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
    pub fn subtract(self, rhs: Self) -> Result<Self> {
        self.plus(rhs.map(&mut |e| -e)?)
    }

    /// conjugate transpose a matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::cmplx;
    /// # use rmatrix_ks::error::Result;
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::num::complex::Complex;
    /// # fn main() -> Result<()> {
    /// let mat: Matrix<Complex<i32>> =
    ///     Matrix::create(1, 2, vec![cmplx!(1, 2), cmplx!(2, 3)])?;
    /// assert_eq!(Matrix::create(2, 1, vec![cmplx!(1, -2), cmplx!(2, -3)])?,
    ///     mat.conjugate_transpose()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn conjugate_transpose(&self) -> Result<Matrix<T>>
    where
        T: Clone,
    {
        let mut transposed = Vec::with_capacity(self.row() * self.column());
        for c in 1..=self.column() {
            for r in 1..=self.row() {
                transposed.push(self.get_element(r, c)?.to_owned().conjugate());
            }
        }
        Matrix::<T>::create(self.column(), self.row(), transposed)
    }

    /// get the trace of a matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let m: Matrix<f32> = Matrix::create(2, 2, vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
    /// assert_eq!(5.0f32, m.trace()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn trace(&self) -> Result<T> {
        let tr = if self.row() == 1 || self.column() == 1 {
            // for vector v.trace() = v.sum()
            self.inner
                .iter()
                .fold(T::zero(), |acc: T, e: &T| acc + e.to_owned())
        } else {
            // else m.trace() = m.diag().sum()
            self.get_diag()?
                .inner
                .iter()
                .cloned()
                .fold(T::zero(), |acc: T, e: &T| acc + e.to_owned())
        };

        Ok(tr)
    }

    /// get the rank of a matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let m = Matrix::<f32>::create(3, 4, vec![
    ///     1.0f32, 2.0f32, 3.0f32, -1.0f32, 4.0f32, 5.0f32, 6.0f32, 2.0f32, 7.0f32, 8.0f32, 9.0f32, 3.0f32,
    /// ])?;
    /// assert_eq!(3usize, m.rank()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn rank(&self) -> Result<usize> {
        let reduced = self.row_eliminate()?.0;

        Ok((1..=self.row())
            .map(|i| Ok(reduced.get_row(i)?.inner))
            .map(|v| -> Result<bool> { Ok(v?.iter().cloned().all(|e| e.is_zero())) })
            .filter(|b| b == &Ok(false))
            .count())
    }

    /// get the submatrix of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let m = Matrix::<f32>::create(3, 4, vec![
    ///     1.0f32, 2.0f32, 3.0f32, -1.0f32, 4.0f32, 5.0f32, 6.0f32, 2.0f32, 7.0f32, 8.0f32, 9.0f32, 3.0f32,
    /// ])?;
    /// assert_eq!(Matrix::create(2, 3, vec![4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32])?,
    ///     m.submatrix(1, 4)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn submatrix(&self, row: usize, col: usize) -> Result<Matrix<T>> {
        let mut submat = Matrix::zeros(self.row() - 1, self.column() - 1)?;
        let mut row_index = 1;
        for r in (1..=self.row()).filter(|e| e != &row) {
            let mut col_index = 1;
            for c in (1..=self.column()).filter(|e| e != &col) {
                submat.set_element(row_index, col_index, self.get_element(r, c)?.to_owned())?;
                col_index = col_index + 1;
            }
            row_index = row_index + 1;
        }
        Ok(submat)
    }

    /// get the adjugate matrix of the matrix
    ///
    /// adj(m) * m = det(m) E
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::Result;
    /// # fn main() -> Result<()> {
    /// let m = Matrix::<f32>::create(2, 2, vec![
    ///     5.0f32, 4.0f32, 4.0f32, 11f32,
    /// ])?;
    /// assert_eq!(Matrix::create(2, 2, vec![11.0f32, -4.0f32, -4.0f32, 5.0f32])?,
    ///     m.adjugate()?);
    /// assert_eq!(m.adjugate()?.times(m.to_owned()),
    ///     Matrix::<f32>::eyes(2, 2)?.muls(m.determinant()?));
    /// # Ok(())
    /// # }
    /// ```
    pub fn adjugate(&self) -> Result<Matrix<T>> {
        if self.row() != self.column() {
            // only square matrix
            Err(Error::IncompatibleShape((self.row(), self.row()), self.dim))
        } else {
            let mut mat = Self::zeros(self.row(), self.column())?;
            for row in 1..=self.row() {
                for col in 1..=self.column() {
                    mat.set_element(
                        row,
                        col,
                        self.submatrix(row, col)?.determinant()?
                            * if (row + col) & 1 == 1 {
                                -T::one()
                            } else {
                                T::one()
                            },
                    )?;
                }
            }
            mat.transpose()
        }
    }

    /// transform the matrix to upper triangle form by rows elimination
    ///
    /// ```rust
    /// # use rmatrix_ks::error::Result;
    /// # use rmatrix_ks::matrix::Matrix;
    /// # fn main() -> Result<()> {
    /// let mat = Matrix::<f32>::create(3, 3, vec![1.0f32, 2.0f32, 4.0f32, 3.0f32, 6.0f32, 8.0f32, 5.0f32, 7.0f32, 9.0f32])?;
    /// let eliminates = mat.row_eliminate()?;
    /// assert_eq!(Matrix::create(3, 3, vec![1.0f32, 2.0f32, 4.0f32, 0.0f32, -3.0f32, -11.0f32, 0.0f32, 0.0f32, -4.0f32])?,
    ///     eliminates.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn row_eliminate(&self) -> Result<(Self, Matrix<T>, T)> {
        let mut reduced = self.to_owned();
        let mut p_all = Matrix::eyes(self.row(), self.row())?;
        let mut lambda = T::one();

        if !(is_upper_triangle_matrix(&reduced) || self.row() < 2) {
            let mut next: usize = 0;
            for index in 1..=(self.row() - 1) {
                // prevent out of boundary
                if index + next > self.column() {
                    break;
                }
                // check pivot
                'check_pivot: while reduced.get_element(index, index + next)?.is_zero() {
                    // find non-zero pivot
                    for above in (index + 1)..=self.row() {
                        //do row exchange
                        if !reduced.get_element(above, index + next)?.is_zero() {
                            let p_change = Matrix::p_change(self.row(), index, above)?;
                            p_all = p_change.to_owned().times(p_all)?;
                            reduced = p_change.times(reduced)?;
                            lambda = -lambda;
                            break 'check_pivot;
                        }
                    }
                    // find next column
                    if index + next < self.column() {
                        next = next + 1;
                    }
                }
                // do eliminate
                let value = reduced.to_owned();
                let pivot = value.get_element(index, index + next)?;
                for above in (index + 1)..=self.row() {
                    // do row add
                    let above_pivot = reduced.get_element(above, index + next)?;
                    // skip zero line
                    if !above_pivot.is_zero() {
                        // warn: for integer, division is non-accuracy, can use rational number
                        let factor = above_pivot.to_owned().ndiv(pivot.to_owned())?;
                        let p_add = Matrix::p_add(self.row(), index, above, -factor)?;
                        p_all = p_add.to_owned().times(p_all)?;
                        reduced = p_add.times(reduced)?;
                    }
                }
            }
        }
        // retuen
        Ok((reduced, p_all, lambda))
    }

    /// determinant of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::error::Result;
    /// # use rmatrix_ks::matrix::Matrix;
    /// # fn main() -> Result<()> {
    /// let mat = Matrix::<f32>::create(3, 3, vec![
    ///     1.0f32, 2.0f32, 4.0f32, 3.0f32, 6.0f32, 8.0f32, 5.0f32, 7.0f32, 9.0f32
    /// ])?;
    /// assert_eq!(-12.0, mat.determinant()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn determinant(&self) -> Result<T> {
        if self.row() != self.column() {
            // only square matrix has determinant
            Err(Error::IncompatibleShape((self.row(), self.row()), self.dim))
        } else {
            // for upper triangle matrix
            // determinant is the production of diagonal
            let (reduced, _, lambda) = self.row_eliminate()?;
            (1..=self.row())
                .map(|index| reduced.get_element(index, index))
                .fold(Ok(T::one()), |acc, e| {
                    e.and_then(|ev| acc.map(|v| v * ev.to_owned()))
                })
                .map(|e| e * lambda) // original matrix row exchange will affect determinant
        }
    }

    /// transform the matrix to standard upper triangle form by rows elimination
    ///
    /// ```rust
    /// # use rmatrix_ks::error::Result;
    /// # use rmatrix_ks::matrix::Matrix;
    /// # fn main() -> Result<()> {
    /// let mat = Matrix::<f32>::create(3, 3,
    ///     vec![1.0f32, 2.0f32, 4.0f32, 3.0f32, 6.0f32, 8.0f32, 5.0f32, 7.0f32, 9.0f32])?;
    /// let reduced = mat.row_reduce()?;
    /// assert_eq!(Matrix::create(3, 3,
    ///     vec![1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32])?,
    ///     reduced.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn row_reduce(&self) -> Result<(Self, Matrix<T>)> {
        // from upper triangle matrix to reduce
        let eliminates = self.row_eliminate()?;
        let mut reduced = eliminates.0;
        // keep record the processes
        let mut p_all = eliminates.1;
        let mut col = 0;
        let mut index = 1;
        // do reduce
        while index <= self.row().min(self.column()) && index + col <= self.column() {
            if reduced.get_element(index, index + col)?.is_zero() {
                // if pivot is zero, skip this column
                col += 1;
            } else {
                // make pivot to one
                // warn: for integer, division is non-accuracy, can use rational number
                let p_smul = Matrix::p_muls(
                    self.row(),
                    index,
                    T::one().ndiv(reduced.get_element(index, index + col)?.to_owned())?,
                )?;
                p_all = p_smul.to_owned().times(p_all)?;
                reduced = p_smul.times(reduced)?;
                // reduce the previous row
                for j in 1..index {
                    if !reduced.get_element(j, index + col)?.is_zero() {
                        let p = reduced
                            .get_element(j, index + col)?
                            .to_owned()
                            .ndiv(reduced.get_element(index, index + col)?.to_owned())?;
                        let p_add = Matrix::p_add(self.row(), index, j, -p)?;
                        p_all = p_add.to_owned().times(p_all)?;
                        reduced = p_add.times(reduced)?;
                    }
                }
                index += 1;
            }
        }
        // return
        Ok((reduced, p_all))
    }

    /// find the inverse of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::error::Result;
    /// # use rmatrix_ks::matrix::Matrix;
    /// # fn main() -> Result<()> {
    /// let mat = Matrix::<f32>::create(2, 2,
    ///     vec![1.4f32, 2.0f32, 3.0f32, -6.7f32])?;
    /// assert_eq!(Matrix::create(2, 2,
    ///     vec![0.43563068f32, 0.13003902f32, 0.19505852f32, -0.09102731f32])?,
    ///     mat.inverse()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn inverse(&self) -> Result<Matrix<T>> {
        // use Gauss-Jordan method
        Ok(self.row_reduce()?.1)
    }
}

/// the simplest format print
impl<T> std::fmt::Display for Matrix<T>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", "\u{007b}")?;
        for r in 1..=self.row() {
            write!(f, "{}", "\u{007b}")?;
            for c in 1..=self.column() {
                if let Ok(element) = self.get_element(r, c) {
                    write!(f, "{}", element)?;
                }
                if c == self.column() {
                    write!(f, "{}", "\u{007d}")?;
                } else {
                    write!(f, ", ")?;
                }
            }
            if r != self.row() {
                write!(f, ", ")?;
            }
        }
        write!(f, "{}", "\u{007d}")
    }
}

/// a matrix equals to itself
impl<T> std::cmp::PartialEq for Matrix<T>
where
    T: Equal,
{
    fn eq(&self, rhs: &Self) -> bool {
        self.equal(rhs)
    }
}

/// access the inner by iter
impl<T> std::iter::IntoIterator for Matrix<T> {
    type Item = T;

    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let length = self.row() * self.column();
        self.inner
            .into_iter()
            .take(length) // for preventing user out of boundary
            .collect::<Vec<_>>()
            .into_iter()
    }
}
