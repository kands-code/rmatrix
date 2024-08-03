//! # Serde
//!
//! read from file and save to file

use crate::matrix::Matrix;

pub fn read<T, const ROW: usize, const COL: usize>(
) -> Result<Matrix<T, ROW, COL>, Box<dyn std::error::Error>>
where
    T: Clone + Default + std::marker::Send + std::marker::Sync + for<'a> serde::Deserializer<'a>,
{
    todo!();
}

pub fn write<T, const ROW: usize, const COL: usize>(
    _mats: Vec<&Matrix<T, ROW, COL>>,
) -> Result<(), Box<dyn std::error::Error>> {
    todo!()
}
