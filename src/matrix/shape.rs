use serde::{Deserialize, Serialize};

use crate::error::RMatrixError;

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
/// size of a matrix
pub struct MatrixShape {
    /// row size
    pub row: usize,
    /// column size
    pub col: usize,
}

impl MatrixShape {
    pub fn new(row: usize, col: usize) -> Result<Self, RMatrixError> {
        if row == 0 || col == 0 {
            Err(RMatrixError::ShapeUnreasonable)
        } else {
            Ok(Self { row, col })
        }
    }

    pub fn vpos(&self, prow: usize, pcol: usize) -> Result<usize, RMatrixError> {
        if prow > self.row || pcol > self.col {
            Err(RMatrixError::OutOfBoundary(prow, pcol))
        } else {
            Ok((prow - 1) * self.col + pcol - 1)
        }
    }
}

impl std::fmt::Display for MatrixShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.row, self.col)
    }
}

impl Default for MatrixShape {
    fn default() -> Self {
        Self { row: 2, col: 2 }
    }
}
