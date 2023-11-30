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

    pub fn row_reduce(&self) -> Result<Self, RMatrixError> {
        let (mut m, _, _) = self.row_eliminate()?;
        let mut bias = 0;
        let size = m.shape.row.min(m.shape.col);
        let mut i = 1;
        while i <= size {
            if !m.get(i, i + bias)?.is_zero() {
                let p = m.get(i, i + bias)?;
                for j in 1..=m.shape.col {
                    m.set(m.get(i, j)? / p, i, j)?;
                }
                for j in 1..i {
                    println!("i = {}, j = {}", i, j);
                    if !m.get(j, i + bias)?.is_zero() {
                        let p = m.get(j, i + bias)? / m.get(i, i + bias)?;
                        m = Matrix::p_add(m.shape.row, -p, i, j)?.times(&m)?;
                    }
                }
                i += 1;
            } else {
                bias += 1;
            }
        }
        Ok(m)
    }

    pub fn det(&self) -> N {
        unimplemented!()
    }
}
