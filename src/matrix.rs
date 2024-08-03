//! # Matrix
//!
//! basic matrix with some common manipulations and features
//!
//! *all indices will start from 1*
//!
//! *do not have any optimization*

use crate::error::MatrixError;
use crate::number::Number;
use crate::utils::is_upper_triangle_matrix;
use crate::vector::times_v;
use crate::vector::VectorC;
use crate::vector::VectorR;

#[cfg(feature = "rand_mat")]
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};

/// matrix type
#[derive(Debug, Clone)]
pub struct Matrix<T, const ROW: usize, const COL: usize> {
    pub(crate) inner: Vec<T>,
}

/// default implementations
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
    /// // {{1i8, 2i8}, {3i8, 4i8}}
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
    /// // {{0i8, 0i8}, {0i8, 0i8}}
    /// let _: Matrix<i8, 2, 2> = Matrix::zeros()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn zeros() -> Result<Self, MatrixError>
    where
        T: Clone + Default,
    {
        Self::create(vec![T::default(); ROW * COL])
    }

    /// get the shape of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
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
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
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
    /// // {{0i8, 0i8}, {0i8, 0i8}}
    /// let mut mat: Matrix<i8, 2, 2> = Matrix::zeros()?;
    /// // {{3i8, 0i8}, {0i8, 0i8}}
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
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// // {{4i8, 5i8, 6i8}}
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
            VectorR::<&T, COL>::create(nth_row)
        }
    }

    /// get the n-th col of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// // {{2i8, 5i8}}
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
            VectorC::<&T, ROW>::create(nth_col)
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
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// // {{1i8, 5i8}}
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
    /// // {{1i8, 2i8, 3i8}, {4i8, 5i8, 6i8}}
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// // {{1i8, 4i8}, {2i8, 5i8}, {3i8, 6i8}}
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

    /// map a function to a matrix
    ///
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
        Matrix::<N, ROW, COL>::create(mapped)
    }
}

/// implementation for matrix which element type is a Number
impl<T, const ROW: usize, const COL: usize> Matrix<T, ROW, COL>
where
    T: Number,
{
    /// create an identity matrix with size row by col
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // {{1.0f32, 0.0f32}, {0.0f32, 1.0f32}}
    /// let _: Matrix<f32, 2, 2> = Matrix::eyes()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn eyes() -> Result<Self, MatrixError> {
        let mut mat = Self::zeros()?;
        for i in 1..=Self::get_edge() {
            mat.set_element(i, i, T::one())?;
        }
        Ok(mat)
    }

    #[cfg(feature = "rand_mat")]
    /// create a random matrix with size row by col
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // need "rand_mat" feature
    /// let _: Matrix<f32, 2, 2> = Matrix::rand()?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn rand() -> Result<Self, MatrixError>
    where
        T: Number,
        Standard: Distribution<T>,
    {
        let mut mat = Self::zeros()?;
        let mut rng = rand::rngs::ThreadRng::default();
        mat.inner = mat.inner.iter().map(|_| rng.gen()).collect();
        Ok(mat)
    }

    /// exchange i row with j row
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let m = Matrix::<f32, 2, 2>::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
    /// let p = Matrix::<f32, 2, 2>::p_change(1, 2)?;
    /// assert_eq!(
    ///     Matrix::<f32, 2, 2>::create(vec![3.0f32, 4.0f32, 1.0f32, 2.0f32])?,
    ///     p.to_owned().times(m.to_owned())?
    /// );
    /// assert_eq!(
    ///     Matrix::<f32, 2, 2>::create(vec![2.0f32, 1.0f32, 4.0f32, 3.0f32])?,
    ///     m.times(p)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn p_change(i: usize, j: usize) -> Result<Matrix<T, ROW, ROW>, MatrixError> {
        let mut mat = Matrix::<T, ROW, ROW>::eyes()?;
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
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let m = Matrix::<f32, 2, 2>::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
    /// let p = Matrix::<f32, 2, 2>::p_muls(1, 2.0f32)?;
    /// assert_eq!(
    ///     Matrix::<f32, 2, 2>::create(vec![2.0f32, 4.0f32, 3.0f32, 4.0f32])?,
    ///     p.to_owned().times(m.to_owned())?
    /// );
    /// assert_eq!(
    ///     Matrix::<f32, 2, 2>::create(vec![2.0f32, 2.0f32, 6.0f32, 4.0f32])?,
    ///     m.times(p)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn p_muls(i: usize, k: T) -> Result<Matrix<T, ROW, ROW>, MatrixError> {
        let mut mat = Matrix::<T, ROW, ROW>::eyes()?;
        mat.set_element(i, i, k)?;
        Ok(mat)
    }

    /// add k times of the i row to the j row
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let m = Matrix::<f32, 2, 2>::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32])?;
    /// let p = Matrix::<f32, 2, 2>::p_add(1, 2, 1.0f32)?;
    /// assert_eq!(
    ///     Matrix::<f32, 2, 2>::create(vec![1.0f32, 2.0f32, 4.0f32, 6.0f32])?,
    ///     p.to_owned().times(m.to_owned())?
    /// );
    /// assert_eq!(
    ///     Matrix::<f32, 2, 2>::create(vec![1.0f32, 3.0f32, 3.0f32, 7.0f32])?,
    ///     m.times(p.transpose()?)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn p_add(i: usize, j: usize, k: T) -> Result<Matrix<T, ROW, ROW>, MatrixError> {
        let mut mat = Matrix::<T, ROW, ROW>::eyes()?;
        mat.set_element(j, i, k)?;
        Ok(mat)
    }

    /// matrix addition
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let mat1: Matrix<f32, 2, 3> = Matrix::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
    /// let mat2: Matrix<f32, 2, 3> = Matrix::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
    /// assert_eq!(Matrix::create(vec![2.0f32, 4.0f32, 6.0f32, 8.0f32, 10.0f32, 12.0f32])?, mat1.plus(mat2)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn plus(self, rhs: Self) -> Result<Self, MatrixError> {
        let sum = self
            .inner
            .iter()
            .zip(rhs.inner.iter())
            .map(|(a, b)| a.to_owned() + b.to_owned())
            .collect::<Vec<_>>();
        Self::create(sum)
    }

    /// matrix addition with scalar
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let mat: Matrix<f32, 2, 3> = Matrix::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
    /// assert_eq!(Matrix::create(vec![2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32])?, mat.adds(1.0f32)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn adds(self, scalar: T) -> Result<Self, MatrixError> {
        self.map(|a| scalar.to_owned() + a)
    }

    /// matrix multiplication
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let a: Matrix<f64, 2, 2> = Matrix::create(vec![1.0f64, 2.0f64, 3.0f64, 4.0f64])?;
    /// let b: Matrix<f64, 2, 2> = Matrix::create(vec![5.0f64, 6.0f64, 7.0f64, 8.0f64])?;
    /// assert_eq!(
    ///     Matrix::<f64, 2, 2>::create(vec![19.0f64, 22.0f64, 43.0f64, 50.0f64])?,
    ///     a.times(b)?
    /// );
    /// # Ok(())
    /// # }
    /// ```
    pub fn times<const SIDE: usize>(
        self,
        rhs: Matrix<T, COL, SIDE>,
    ) -> Result<Matrix<T, ROW, SIDE>, MatrixError> {
        let mut product = Matrix::<T, ROW, SIDE>::create(vec![T::zero(); ROW * SIDE])?;
        for c in 1..=COL {
            product = product
                .plus(times_v(
                    self.get_col(c)?.map(|e| e.to_owned())?,
                    rhs.get_row(c)?.map(|e| e.to_owned())?,
                )?)?
                .to_owned();
        }
        Ok(product)
    }

    /// matrix multiplication with scalar
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let mat: Matrix<f32, 2, 3> = Matrix::create(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])?;
    /// assert_eq!(Matrix::create(vec![2.0f32, 4.0f32, 6.0f32, 8.0f32, 10.0f32, 12.0f32])?, mat.muls(2.0f32)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn muls(self, scalar: T) -> Result<Self, MatrixError> {
        self.map(|a| scalar.to_owned() * a)
    }

    /// matrix subtraction
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let a: Matrix<f64, 2, 2> = Matrix::create(vec![1.0f64, 2.0f64, 3.0f64, 4.0f64])?;
    /// let b: Matrix<f64, 2, 2> = Matrix::create(vec![5.0f64, 6.0f64, 7.0f64, 8.0f64])?;
    /// assert_eq!(
    ///     Matrix::<f64, 2, 2>::create(vec![4.0f64, 4.0f64, 4.0f64, 4.0f64])?,
    ///     b.subtract(a)?
    /// );
    /// # Ok(())
    /// # }
    pub fn subtract(self, rhs: Self) -> Result<Self, MatrixError> {
        self.plus(rhs.map(|e| -e)?)
    }

    /// get the trace of a matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let m: Matrix<f64, 2, 2> = Matrix::create(vec![1.0f64, 2.0f64, 3.0f64, 4.0f64])?;
    /// assert_eq!(5.0f64, m.trace()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn trace(&self) -> Result<T, MatrixError>
    where
        [(); Self::get_edge()]:,
    {
        if ROW == 1 || COL == 1 {
            // for vector v.trace() = v.sum()
            Ok(self
                .inner
                .iter()
                .fold(T::zero(), |acc: T, e: &T| acc + e.to_owned()))
        } else {
            // else m.trace() = m.diag().sum()
            Ok(self
                .get_diag()?
                .inner
                .iter()
                .cloned()
                .fold(T::zero(), |acc: T, e: &T| acc + e.to_owned()))
        }
    }

    /// get the rank of a matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let m = Matrix::<f32, 3, 4>::create(vec![
    ///     1.0f32, 2.0f32, 3.0f32, -1.0f32, 4.0f32, 5.0f32, 6.0f32, 2.0f32, 7.0f32, 8.0f32, 9.0f32, 3.0f32,
    /// ])?;
    /// assert_eq!(3usize, m.rank()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn rank(&self) -> Result<usize, MatrixError> {
        let reduced = self.row_eliminate()?.0;
        Ok((1..=ROW)
            .rev()
            .map(|i| Ok(reduced.get_row(i)?.inner))
            .map(|v| -> Result<bool, MatrixError> { Ok(v?.iter().cloned().all(|e| e.is_zero())) })
            .filter(|b| b == &Ok(false))
            .count())
    }

    /// get the submatrix of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let m = Matrix::<f32, 3, 4>::create(vec![
    ///     1.0f32, 2.0f32, 3.0f32, -1.0f32, 4.0f32, 5.0f32, 6.0f32, 2.0f32, 7.0f32, 8.0f32, 9.0f32, 3.0f32,
    /// ])?;
    /// assert_eq!(Matrix::create(vec![4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32])?,
    ///     m.submatrix(1, 4)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn submatrix(
        &self,
        row: usize,
        col: usize,
    ) -> Result<Matrix<T, { ROW - 1 }, { COL - 1 }>, MatrixError> {
        let mut submat = Matrix::zeros()?;
        let mut row_index = 1;
        for r in (1..=ROW).filter(|e| e != &row) {
            let mut col_index = 1;
            for c in (1..=COL).filter(|e| e != &col) {
                submat.set_element(row_index, col_index, self.get_element(r, c)?.to_owned())?;
                col_index = col_index + 1;
            }
            row_index = row_index + 1;
        }
        Ok(submat)
    }

    /// get the cofactor matrix of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let m = Matrix::<f32, 3, 4>::create(vec![
    ///     1.0f32, 2.0f32, 3.0f32, -1.0f32, 4.0f32, 5.0f32, 6.0f32, 2.0f32, 7.0f32, 8.0f32, 9.0f32, 3.0f32,
    /// ])?;
    /// assert_eq!(Matrix::create(vec![-4.0f32, -5.0f32, -6.0f32, -7.0f32, -8.0f32, -9.0f32])?,
    ///     m.cofactor(1, 4)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn cofactor(
        &self,
        row: usize,
        col: usize,
    ) -> Result<Matrix<T, { ROW - 1 }, { COL - 1 }>, MatrixError> {
        let cofactor = self.submatrix(row, col)?;
        let cofactor = cofactor.muls(if (row + col) & 1 == 1 {
            -T::one()
        } else {
            T::one()
        })?;
        Ok(cofactor)
    }

    /// get the adjugate matrix of the matrix
    ///
    /// adj(m) * m = det(m) E
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// let m = Matrix::<f32, 2, 2>::create(vec![
    ///     5.0f32, 4.0f32, 4.0f32, 11f32,
    /// ])?;
    /// assert_eq!(Matrix::create(vec![11.0f32, -4.0f32, -4.0f32, 5.0f32])?,
    ///     m.adjugate()?);
    /// assert_eq!(m.adjugate()?.times(m.to_owned()),
    ///     Matrix::<f32, 2, 2>::eyes()?.muls(m.determinant()?));
    /// # Ok(())
    /// # }
    /// ```
    pub fn adjugate(&self) -> Result<Matrix<T, COL, ROW>, MatrixError>
    where
        [(); ROW - 1]:,
        [(); COL - 1]:,
    {
        if ROW != COL {
            // only square matrix
            Err(MatrixError::IncompatibleShape((ROW, ROW), (ROW, COL)))
        } else {
            let mut mat = Self::zeros()?;
            for row in 1..=ROW {
                for col in 1..=COL {
                    mat.set_element(row, col, self.cofactor(row, col)?.determinant()?)?;
                }
            }
            mat.transpose()
        }
    }

    /// transform the matrix to upper triangle form by rows elimination
    ///
    /// ```rust
    /// # use rmatrix_ks::error::MatrixError;
    /// # use rmatrix_ks::matrix::Matrix;
    /// # fn main() -> Result<(), MatrixError> {
    /// let mat = Matrix::<f32, 3, 3>::create(vec![1.0f32, 2.0f32, 4.0f32, 3.0f32, 6.0f32, 8.0f32, 5.0f32, 7.0f32, 9.0f32])?;
    /// let eliminates = mat.row_eliminate()?;
    /// assert_eq!(Matrix::create(vec![1.0f32, 2.0f32, 4.0f32, 0.0f32, -3.0f32, -11.0f32, 0.0f32, 0.0f32, -4.0f32])?,
    ///     eliminates.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn row_eliminate(&self) -> Result<(Self, Self, T), MatrixError> {
        let mut reduced = self.to_owned();
        let mut p_all = Self::eyes()?;
        let mut lambda = T::one();

        if !is_upper_triangle_matrix(&reduced)? {
            let mut col: usize = 0;
            for index in 1..=(ROW - 1) {
                // prevent out of boundary
                if index + col > COL {
                    break;
                }
                // check pivot
                'check_pivot: while reduced.get_element(index, index + col)?.is_zero() {
                    // find non-zero pivot
                    for above in (index + 1)..=ROW {
                        //do row exchange
                        if !reduced.get_element(above, index + col)?.is_zero() {
                            let p_change = Self::p_change(index, above)?;
                            p_all = p_change.to_owned().times(p_all)?;
                            reduced = p_change.times(reduced)?;
                            lambda = -lambda;
                            break 'check_pivot;
                        }
                    }
                    // find next column
                    if index + col < COL {
                        col = col + 1;
                    }
                }
                // do eliminate
                let value = reduced.to_owned();
                let pivot = value.get_element(index, index + col)?;
                for above in (index + 1)..=ROW {
                    // do row add
                    let above_pivot = reduced.get_element(above, index + col)?;
                    // *warn*: for integer, division is non-accuracy, can use rational number
                    let factor = above_pivot.to_owned().ndiv(pivot.to_owned())?;
                    let p_add = Self::p_add(index, above, -factor)?;
                    p_all = p_add.to_owned().times(p_all)?;
                    reduced = p_add.times(reduced)?;
                }
            }
        }
        // retuen
        Ok((reduced, p_all, lambda))
    }

    /// determinant of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::error::MatrixError;
    /// # use rmatrix_ks::matrix::Matrix;
    /// # fn main() -> Result<(), MatrixError> {
    /// let mat = Matrix::<f32, 3, 3>::create(vec![
    ///     1.0f32, 2.0f32, 4.0f32, 3.0f32, 6.0f32, 8.0f32, 5.0f32, 7.0f32, 9.0f32
    /// ])?;
    /// assert_eq!(-12.0, mat.determinant()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn determinant(&self) -> Result<T, MatrixError> {
        if ROW != COL {
            // only square matrix has determinant
            Err(MatrixError::IncompatibleShape((ROW, ROW), (ROW, COL)))
        } else {
            // for upper triangle matrix
            // determinant is the production of diagonal
            let (reduced, _, lambda) = self.row_eliminate()?;
            (1..=ROW)
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
    /// # use rmatrix_ks::error::MatrixError;
    /// # use rmatrix_ks::matrix::Matrix;
    /// # fn main() -> Result<(), MatrixError> {
    /// let mat = Matrix::<f32, 3, 3>::create(
    ///     vec![1.0f32, 2.0f32, 4.0f32, 3.0f32, 6.0f32, 8.0f32, 5.0f32, 7.0f32, 9.0f32])?;
    /// let reduced = mat.row_reduce()?;
    /// assert_eq!(Matrix::create(
    ///     vec![1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32])?,
    ///     reduced.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn row_reduce(&self) -> Result<(Self, Self), MatrixError> {
        // from upper triangle matrix to reduce
        let eliminates = self.row_eliminate()?;
        let mut reduced = eliminates.0;
        // keep record the processes
        let mut p_all = eliminates.1;
        let mut col = 0;
        let mut index = 1;
        // do reduce
        while index <= Self::get_edge() && index + col <= COL {
            if reduced.get_element(index, index + col)?.is_zero() {
                // if pivot is zero, skip this column
                col += 1;
            } else {
                // make pivot to one
                // *warn*: for integer, division is non-accuracy, can use rational number
                let p_smul = Self::p_muls(
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
                        let p_add = Self::p_add(index, j, -p)?;
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
    /// # use rmatrix_ks::error::MatrixError;
    /// # use rmatrix_ks::matrix::Matrix;
    /// # fn main() -> Result<(), MatrixError> {
    /// let mat = Matrix::<f32, 2, 2>::create(
    ///     vec![1.4f32, 2.0f32, 3.0f32, -6.7f32])?;
    /// assert_eq!(Matrix::create(
    ///     vec![0.43563068f32, 0.13003902f32, 0.19505852f32, -0.09102731f32])?,
    ///     mat.inverse()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn inverse(&self) -> Result<Self, MatrixError> {
        // Gauss-Jordan method
        Ok(self.row_reduce()?.1)
    }
}

/// the simplest format print
impl<T, const ROW: usize, const COL: usize> std::fmt::Display for Matrix<T, ROW, COL>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for r in 1..=ROW {
            write!(f, "{{")?;
            for c in 1..=COL {
                if let Ok(element) = self.get_element(r, c) {
                    write!(f, "{}", element)?;
                }
                if c == COL {
                    write!(f, "}}")?;
                } else {
                    write!(f, ", ")?;
                }
            }
            if r != ROW {
                write!(f, ", ")?;
            }
        }
        write!(f, "}}")
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
