//! attribute of matrices

use crate::{error::RMatrixError, matrix::Matrix, number::Number};

impl<N: Number> Matrix<N> {
    /// shape of a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// # use rmatrix_ks::matrix::Matrix;
    /// # use rmatrix_ks::complex::Complex;
    /// # use rmatrix_ks::error::RMatrixError;
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
        let mut t = N::default();
        for i in 1..=self.shape.row.min(self.shape.col) {
            t = t + self.get(i, i)?;
        }
        Ok(t)
    }

    pub fn inverse(&self) -> Result<Self, RMatrixError> {
        if self.shape.row != self.shape.col {
            Err(RMatrixError::MatrixNotSquare)
        } else {
            Ok(self.row_reduce()?.1)
        }
    }

    pub fn rank(&self) -> Result<usize, RMatrixError> {
        let (m, _, _) = self.row_eliminate()?;
        Ok((1..=m.shape.row)
            .map(|i| m.get_row(i).unwrap())
            .map(|r| r.iter().any(|v| !v.is_zero()))
            .filter(|&v| v)
            .count())
    }
}
