//! # Matrix
//!
//! basic matrix with some common manipulations and features
//!
//! *all indices will start from 1*
//!
//! *does not have any optimization*

use crate::error::MatrixError;
use crate::vector::VectorC;
use crate::vector::VectorR;

/// matrix type
#[derive(Debug, Clone)]
pub struct Matrix<T, const ROW: usize, const COL: usize> {
    pub(crate) inner: Vec<T>,
}

impl<T, const ROW: usize, const COL: usize> Matrix<T, ROW, COL> {
    /// create a matrix with size row by col
    ///
    /// **note: this will consume the original vector**
    ///
    /// this is the default constructor of matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // [[1i8, 2i8], [3i8, 4i8]]
    /// let _: Matrix<i8, 2, 2> = Matrix::create(vec![1, 2, 3, 4])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn create(data: Vec<T>) -> Result<Self, MatrixError> {
        if data.len() < ROW * COL {
            Err(MatrixError::IncompatibleSizeError((ROW, COL), data.len()))
        } else {
            Ok(Matrix::<T, ROW, COL> { inner: data })
        }
    }

    /// create a zero matrix with size row by col
    ///
    /// this constructor is similar to Default, but the return value is Result
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // [[0i8, 0i8], [0i8, 0i8]]
    /// let _: Matrix<i8, 2, 2> = Matrix::zeros()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn zeros() -> Result<Matrix<T, ROW, COL>, MatrixError>
    where
        T: Clone + Default,
    {
        Matrix::<T, ROW, COL>::create(vec![T::default(); ROW * COL])
    }

    /// get the shape of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // [[1i8, 2i8, 3i8], [4i8, 5i8, 6i8]]
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!((2, 3), mat.dimensions());
    /// # Ok(())
    /// # }
    /// ```
    pub fn dimensions(&self) -> (usize, usize) {
        (ROW, COL)
    }

    /// convert index into inner index
    pub(crate) const fn to_inner_index(row: usize, col: usize) -> usize {
        (row - 1) * COL + col - 1
    }

    /// get the value of the specific position of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // [[1i8, 2i8, 3i8], [4i8, 5i8, 6i8]]
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!(&5, mat.get_element(2, 2)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_element(&self, row: usize, col: usize) -> Result<&T, MatrixError> {
        match self.inner.get(Self::to_inner_index(row, col)) {
            Some(element) => Ok(element),
            None => Err(MatrixError::OutOfBoundary(row, col)),
        }
    }

    /// set the value of the specific position of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // [[0i8, 0i8], [0i8, 0i8]]
    /// let mut mat: Matrix<i8, 2, 2> = Matrix::zeros()?;
    /// // [[3i8, 0i8], [0i8, 0i8]]
    /// mat.set_element(1, 1, 3i8)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_element(&mut self, row: usize, col: usize, replace: T) -> Result<(), MatrixError> {
        match self.inner.get_mut(Self::to_inner_index(row, col)) {
            Some(element) => {
                *element = replace;
                Ok(())
            }
            None => Err(MatrixError::OutOfBoundary(row, col)),
        }
    }

    /// get the n-th row of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // [[1i8, 2i8, 3i8], [4i8, 5i8, 6i8]]
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// // [[4i8, 5i8, 6i8]]
    /// let _ = mat.get_row(2)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_row(&self, row: usize) -> Result<VectorR<&T, COL>, MatrixError> {
        if row > ROW {
            Err(MatrixError::OutOfBoundary(row, 1))
        } else {
            let mut nth_row = Vec::with_capacity(COL);
            for c in 1..=COL {
                nth_row.push(self.get_element(row, c)?);
            }
            Matrix::create(nth_row)
        }
    }

    /// get the n-th col of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // [[1i8, 2i8, 3i8], [4i8, 5i8, 6i8]]
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// // [[2i8, 5i8]]
    /// let _ = mat.get_col(2)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_col(&self, col: usize) -> Result<VectorC<&T, ROW>, MatrixError> {
        if col > COL {
            Err(MatrixError::OutOfBoundary(1, col))
        } else {
            let mut nth_col = Vec::with_capacity(COL);
            for r in 1..=ROW {
                nth_col.push(self.get_element(r, col)?);
            }
            Matrix::create(nth_col)
        }
    }

    /// get the shorter edge between `ROW` and `COL`
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # fn main() {
    /// assert_eq!(2, Matrix::<i8, 2, 3>::get_edge());
    /// # }
    /// ```
    pub const fn get_edge() -> usize {
        if ROW > COL {
            COL
        } else {
            ROW
        }
    }

    /// get the main diagonal of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // [[1i8, 2i8, 3i8], [4i8, 5i8, 6i8]]
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// // [[1i8, 5i8]]
    /// let _ = mat.get_diag()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_diag(&self) -> Result<VectorC<&T, { Self::get_edge() }>, MatrixError> {
        let edge: usize = Self::get_edge();
        let mut diag = Vec::with_capacity(edge);
        for i in 1..=edge {
            diag.push(self.get_element(i, i)?);
        }
        VectorC::<&T, { Self::get_edge() }>::create(diag)
    }

    /// transpose a matrix
    ///
    /// **note: this will consume the original matrix**
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // [[1i8, 2i8, 3i8], [4i8, 5i8, 6i8]]
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// // [[1i8, 4i8], [2i8, 5i8], [3i8, 6i8]]
    /// let _: Matrix<i8, 3, 2> = mat.transpose()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn transpose(self) -> Result<Matrix<T, COL, ROW>, MatrixError>
    where
        T: Clone,
    {
        let mut transposed = Vec::with_capacity(self.inner.capacity());
        for c in 1..=COL {
            for r in 1..=ROW {
                transposed.push(self.get_element(r, c)?.to_owned());
            }
        }
        Matrix::<T, COL, ROW>::create(transposed)
    }

    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// let zero: Matrix<i8, 2, 3> = Matrix::create(vec![0i8; 6])?;
    /// assert_eq!(zero, mat.map(|e| e * 0)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn map<N>(&self, mut f: impl FnMut(T) -> N) -> Result<Matrix<N, ROW, COL>, MatrixError>
    where
        T: Clone,
    {
        let mapped = self
            .inner
            .iter()
            .map(|e| f(e.to_owned()))
            .collect::<Vec<_>>();
        Matrix::create(mapped)
    }
}

/// the simplest format print
impl<T, const ROW: usize, const COL: usize> std::fmt::Display for Matrix<T, ROW, COL>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for r in 1..=ROW {
            write!(f, "[")?;
            for c in 1..=COL {
                if let Ok(element) = self.get_element(r, c) {
                    write!(f, "{:#?}", element)?;
                }
                if c == COL {
                    write!(f, "]")?;
                } else {
                    write!(f, ", ")?;
                }
            }
            if r != ROW {
                write!(f, ", ")?;
            }
        }
        write!(f, "]")
    }
}

/// a matrix equals to itself
impl<T, const ROW: usize, const COL: usize> std::cmp::PartialEq for Matrix<T, ROW, COL>
where
    T: std::cmp::PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

#[cfg(test)]
mod inner_test {
    use crate::matrix::Matrix;

    #[test]
    fn test_to_inner_index() {
        assert_eq!(1, Matrix::<i8, 2, 3>::to_inner_index(1, 2));
    }
}
