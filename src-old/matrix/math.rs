//! some math operation of matrices

use crate::{error::RMatrixError, matrix::Matrix, number::Number};

impl<N: Number> Matrix<N> {
    pub fn row_eliminate(&self) -> Result<(Self, Self, N), RMatrixError> {
        let mut m = self.clone();
        let mut factor = N::one();
        let size = m.shape.row.min(m.shape.col);
        let mut np = Matrix::eyes(size, size)?;
        let mut bias = 0;
        let mut i = 1;
        while i + bias < size {
            // check pivot
            while i + bias <= m.shape.col && m.get(i, i + bias)?.is_zero() {
                // find non-zero row
                let mut changed = false;
                for k in (i + 1)..=m.shape.row {
                    if !m.get(k, i + bias)?.is_zero() {
                        // do row exchange
                        let p_change = Matrix::p_change(self.shape.row, i, k)?;
                        np = p_change.times(&np)?;
                        m = p_change.times(&m)?;
                        factor = factor * N::neg_one();
                        changed = true;
                    }
                }
                if !changed {
                    // skip this column
                    bias += 1;
                }
            }
            // do eliminate
            let pivot = m.get(i, i + bias)?;
            for j in (i + 1)..=m.shape.row {
                let p = m.get(j, i + bias)? / pivot;
                let p_add = Matrix::p_add(self.shape.row, -p, i, j)?;
                np = p_add.times(&np)?;
                m = p_add.times(&m)?;
            }
            // do next row
            i += 1;
        }
        Ok((m, np, factor))
    }

    pub fn row_reduce(&self) -> Result<(Self, Self), RMatrixError> {
        let (mut m, mut np, _) = self.row_eliminate()?;
        let mut bias = 0;
        let size = m.shape.row.min(m.shape.col);
        let mut i = 1;
        while i <= size && i + bias <= m.shape.col {
            if !m.get(i, i + bias)?.is_zero() {
                let p = m.get(i, i + bias)?;
                let p_smul = Matrix::p_smul(m.shape.row, N::one() / p, i)?;
                np = p_smul.times(&np)?;
                m = p_smul.times(&m)?;
                for j in 1..i {
                    if !m.get(j, i + bias)?.is_zero() {
                        let p = m.get(j, i + bias)? / m.get(i, i + bias)?;
                        let p_add = Matrix::p_add(m.shape.row, -p, i, j)?;
                        np = p_add.times(&np)?;
                        m = p_add.times(&m)?;
                    }
                }
                i += 1;
            } else {
                bias += 1;
            }
        }
        Ok((m, np))
    }

    pub fn det(&self) -> Result<N, RMatrixError> {
        if self.shape.row != self.shape.col {
            Err(RMatrixError::MatrixNotSquare)
        } else {
            let (m, _, a) = self.row_eliminate()?;
            let mut res = N::one();
            for i in 1..=m.shape.row {
                res = res * m.get(i, i)?;
            }
            Ok(res * a)
        }
    }
}
