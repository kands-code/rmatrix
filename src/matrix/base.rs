use crate::{error::RMatrixError, matrix::Matrix, types::Number};

impl<N: Number> Matrix<N> {
    pub fn transpose(&self) -> Result<Self, RMatrixError> {
        let mut m = Matrix::zeros(self.shape.col, self.shape.row)?;
        for i in 1..=self.shape.row {
            for j in 1..=self.shape.col {
                m.set(self.get(i, j)?, j, i)?;
            }
        }
        Ok(m)
    }

    pub fn plus(&self, rhs: &Self) -> Result<Self, RMatrixError> {
        if self.shape != rhs.shape {
            Err(RMatrixError::ShapeInconsistent(
                rhs.dimensions(),
                self.dimensions(),
            ))
        } else {
            let mut m = Self::zeros(self.shape.row, rhs.shape.col)?;
            for i in 0..self.shape.row * rhs.shape.col {
                m.data[i] = self.data[i] + rhs.data[i];
            }
            Ok(m)
        }
    }

    pub fn times(&self, rhs: &Self) -> Result<Self, RMatrixError> {
        if self.shape.col != rhs.shape.row {
            Err(RMatrixError::ShapeInconsistent(
                rhs.dimensions(),
                self.dimensions(),
            ))
        } else {
            let mut m = Self::zeros(self.shape.row, rhs.shape.col)?;
            for i in 1..=self.shape.col {
                m = m.plus(&Self::outer(self.get_col(i)?, rhs.get_row(i)?)?)?;
            }
            Ok(m)
        }
    }

    pub fn smul(&self, k: N) -> Result<Self, RMatrixError> {
        let mut m = Self::zeros(self.shape.row, self.shape.col)?;
        for i in 0..self.data.len() {
            m.data[i] = self.data[i] * k;
        }
        Ok(m)
    }

    pub fn subtract(&self, rhs: &Self) -> Result<Self, RMatrixError> {
        if self.shape != rhs.shape {
            Err(RMatrixError::ShapeInconsistent(
                self.dimensions(),
                rhs.dimensions(),
            ))
        } else {
            let mut m = Self::zeros(self.shape.row, rhs.shape.col)?;
            for i in 0..self.shape.row * rhs.shape.col {
                m.data[i] = self.data[i] - rhs.data[i];
            }
            Ok(m)
        }
    }
}
