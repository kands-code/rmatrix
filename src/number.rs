//! # Number
//!
//! define the basic number typeclass

use crate::error::MatrixError;
use crate::matrix::Matrix;
use crate::vector::times_v;

/// number typeclass
///
/// # Example
///
/// ```rust,ignore
/// impl Number for i8 {
///     fn one() -> Self {
///         1i8
///     }
///
///     fn ndiv(self, rhs: Self) -> Result<Self, MatrixError> {
///         if rhs.is_zero() {
///             Err(MatrixError::DividedByZero)
///         } else {
///             Ok(self / rhs)
///         }
///     }
///
///     fn is_zero(&self) -> bool {
///         self == &0i8
///     }
/// }
/// ```
pub trait Number
where
    Self: std::clone::Clone
        + std::cmp::PartialEq
        + std::default::Default
        + std::fmt::Debug
        + std::iter::Sum
        + std::ops::Add<Output = Self>
        + std::ops::Sub<Output = Self>
        + std::ops::Mul<Output = Self>
        + std::ops::Neg<Output = Self>
        + std::str::FromStr,
{
    /// the ONE of the number type
    fn one() -> Self;

    /// the ZERO of the number type
    fn zero() -> Self {
        Self::default()
    }

    /// normal division with zero test
    fn ndiv(self, rhs: Self) -> Result<Self, MatrixError>;

    /// check whether a ZERO
    fn is_zero(&self) -> bool;

    /// check whether a ONE
    fn is_one(&self) -> bool {
        (self.to_owned() - Self::one()).is_zero()
    }
}

impl Number for i8 {
    fn one() -> Self {
        1i8
    }

    fn ndiv(self, rhs: Self) -> Result<Self, MatrixError> {
        if rhs.is_zero() {
            Err(MatrixError::DividedByZero)
        } else {
            Ok(self / rhs)
        }
    }

    fn is_zero(&self) -> bool {
        self == &0i8
    }
}

impl Number for f64 {
    fn one() -> Self {
        1.0f64
    }

    fn ndiv(self, rhs: Self) -> Result<Self, MatrixError> {
        if rhs.is_zero() {
            Err(MatrixError::DividedByZero)
        } else {
            Ok(self / rhs)
        }
    }

    fn is_zero(&self) -> bool {
        self < &f64::EPSILON
    }
}

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
    /// // [[1i8, 0i8], [0i8, 1i8]]
    /// let _: Matrix<i8, 2, 2> = Matrix::eyes()?;
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

    /// exchange i row with j row
    ///
    /// ```rust
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::error::MatrixError;
    /// # fn main() -> Result<(), MatrixError> {
    /// // [[1i8, 0i8], [0i8, 1i8]]
    /// let m = Matrix::<i8, 2, 2>::create(vec![1, 2, 3, 4])?;
    /// let p = Matrix::<i8, 2, 2>::p_change(1, 2)?;
    /// assert_eq!(
    ///     Matrix::<i8, 2, 2>::create(vec![3, 4, 1, 2])?,
    ///     p.clone().times(m.clone())?
    /// );
    /// assert_eq!(
    ///     Matrix::<i8, 2, 2>::create(vec![2, 1, 4, 3])?,
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
    /// // [[1i8, 0i8], [0i8, 1i8]]
    /// let m = Matrix::<i8, 2, 2>::create(vec![1, 2, 3, 4])?;
    /// let p = Matrix::<i8, 2, 2>::p_muls(1, 2)?;
    /// assert_eq!(
    ///     Matrix::<i8, 2, 2>::create(vec![2, 4, 3, 4])?,
    ///     p.clone().times(m.clone())?
    /// );
    /// assert_eq!(
    ///     Matrix::<i8, 2, 2>::create(vec![2, 2, 6, 4])?,
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
    /// // [[1i8, 0i8], [0i8, 1i8]]
    /// let m = Matrix::<i8, 2, 2>::create(vec![1, 2, 3, 4])?;
    /// let p = Matrix::<i8, 2, 2>::p_add(1, 2, 1)?;
    /// assert_eq!(
    ///     Matrix::<i8, 2, 2>::create(vec![1, 2, 4, 6])?,
    ///     p.clone().times(m.clone())?
    /// );
    /// assert_eq!(
    ///     Matrix::<i8, 2, 2>::create(vec![1, 3, 3, 7])?,
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
    /// let mat1: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// let mat2: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!(Matrix::create(vec![2, 4, 6, 8, 10, 12])?, mat1.plus(mat2)?);
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
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!(Matrix::create(vec![2, 3, 4, 5, 6, 7])?, mat.adds(1)?);
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
    /// let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    /// assert_eq!(Matrix::create(vec![2, 4, 6, 8, 10, 12])?, mat.muls(2)?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn muls(self, scalar: T) -> Result<Self, MatrixError> {
        self.map(|a| scalar.to_owned() * a)
    }

    /// matrix subtraction
    ///
    /// a - b = a + (-b)
    pub fn subtract(self, rhs: Self) -> Result<Self, MatrixError> {
        self.plus(rhs.map(|e| -e)?)
    }
}

#[cfg(test)]
mod inner_test {
    use crate::number::Number;
    use rand::rngs::ThreadRng;
    use rand::Rng;

    #[test]
    fn test_f64_is_zero_or_one() {
        assert!((0.0f64).is_zero());
        let mut rng = ThreadRng::default();
        for _ in 0..1024 {
            let float64: f64 = rng.gen();
            // a - a = 0
            assert!((float64 - float64).is_zero());
            if !float64.is_zero() {
                // a / a = 1
                assert!((float64 / float64).is_one());
            }
        }
    }
}
