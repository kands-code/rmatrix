//! attribute of a matrix

use crate::{error::RMatrixError, matrix::Matrix, number::Number};

impl<N: Number> Matrix<N> {
    /// shape of a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// # use rmatrix::matrix::Matrix;
    /// # use rmatrix::complex::Complex;
    /// # use rmatrix::error::RMatrixError;
    /// # fn main() -> Result<(), RMatrixError> {
    /// let m = Matrix::<Complex>::zeros(2, 2)?;
    /// assert_eq!(m.dimensions(), (2, 2));
    /// #     Ok(())
    /// # }
    /// ```
    pub fn dimensions(&self) -> (usize, usize) {
        (self.shape.row, self.shape.col)
    }

    pub fn get(&self, prow: usize, pcol: usize) -> Result<N, RMatrixError> {
        Ok(self.data[self.shape.vpos(prow, pcol)?].to_owned())
    }

    pub fn get_row(&self, r: usize) -> Result<Vec<N>, RMatrixError> {
        if r > self.shape.row || r == 0 {
            Err(RMatrixError::OutOfRowBoundary(r))
        } else {
            let mut mrow = Vec::new();
            for c in 1..=self.shape.col {
                mrow.push(self.get(r, c)?);
            }
            Ok(mrow)
        }
    }

    pub fn get_col(&self, c: usize) -> Result<Vec<N>, RMatrixError> {
        if c > self.shape.col || c == 0 {
            Err(RMatrixError::OutOfColumnBoundary(c))
        } else {
            let mut mcol = Vec::new();
            for r in 1..=self.shape.row {
                mcol.push(self.get(r, c)?);
            }
            Ok(mcol)
        }
    }

    pub fn set(&mut self, elem: N, prow: usize, pcol: usize) -> Result<(), RMatrixError> {
        self.data[self.shape.vpos(prow, pcol)?] = elem;
        Ok(())
    }

    pub fn tr(&self) -> Result<N, RMatrixError> {
        let mut t = Default::default();
        for i in 1..=self.shape.row.min(self.shape.col) {
            t = t + self.get(i, i)?;
        }
        Ok(t)
    }
}
