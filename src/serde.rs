//! # Serde
//!
//! read from file and save to file

use crate::matrix::Matrix;

#[cfg(feature = "serde_mat")]
impl<T, const ROW: usize, const COL: usize> Matrix<T, ROW, COL> {
    pub fn read() {
        todo!()
    }

    pub fn write() {
        todo!()
    }
}
