//! # Serde
//!
//! read from file and save to file

use std::io::{Read, Write};

use crate::{error::MatrixError, matrix::Matrix};

/// read matrix from json file with specific index
///
/// index start from 1
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::serde::{read, write};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mat1 = Matrix::<f64, 3, 3>::create(vec![1.0, 2.0, 3.0, 4.0, 8.0, 7.0, 5.0, 9.0, 6.0])?;
/// let mat2 = Matrix::<f64, 3, 3>::create(vec![1.0, 3.0, 2.0, 4.0, 8.0, 7.0, 5.0, 9.0, 6.0])?;
/// write(&[&mat1, &mat2], std::path::Path::new("data/test.json"))?;
/// assert_eq!(
///     mat1,
///     read::<f64, 3, 3>(std::path::Path::new("data/test.json"), 1)?
/// );
/// assert_eq!(
///     mat2,
///     read::<f64, 3, 3>(std::path::Path::new("data/test.json"), 2)?
/// );
/// # Ok(())
/// # }
/// ```
pub fn read<T, const ROW: usize, const COL: usize>(
    path: &std::path::Path,
    index: usize,
) -> Result<Matrix<T, ROW, COL>, Box<dyn std::error::Error>>
where
    T: Clone + for<'a> serde::Deserialize<'a>,
{
    let mut file = std::fs::File::open(path)?;
    let mut mats_string = String::new();
    file.read_to_string(&mut mats_string)?;
    let mats: Vec<Matrix<T, ROW, COL>> = serde_json::from_str(&mats_string)?;
    match mats.get(index - 1) {
        Some(mat) => Ok(mat.to_owned()),
        None => Err(Box::new(MatrixError::OutOfLength(mats.len(), index))),
    }
}

/// store matrices into json file
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::serde::{read, write};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mat1 = Matrix::<f64, 3, 3>::create(vec![1.0, 2.0, 3.0, 4.0, 8.0, 7.0, 5.0, 9.0, 6.0])?;
/// let mat2 = Matrix::<f64, 3, 3>::create(vec![1.0, 3.0, 2.0, 4.0, 8.0, 7.0, 5.0, 9.0, 6.0])?;
/// write(&[&mat1, &mat2], std::path::Path::new("data/test.json"))?;
/// # Ok(())
/// # }
/// ```
pub fn write<T, const ROW: usize, const COL: usize>(
    mats: &[&Matrix<T, ROW, COL>],
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>>
where
    T: serde::Serialize,
{
    let mut file = std::fs::File::create(path)?;
    let mats_string = serde_json::to_string(&mats)?;
    file.write(mats_string.as_bytes())?;
    Ok(())
}
