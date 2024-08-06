//! # Serde
//!
//! read from file and save to file

use crate::error::Error;
use crate::error::Result;
use crate::matrix::Matrix;

use std::io::Read;
use std::io::Write;

#[cfg(feature = "serde_mat")]
#[doc(cfg(feature = "serde_mat"))]
/// read matrix from json file with specific index
///
/// index start from 1
///
/// ```rust
/// # use rmatrix_ks::error::Result;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::serde::{read, write};
/// # fn main() -> Result<()> {
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
) -> Result<Matrix<T, ROW, COL>>
where
    T: Clone + for<'a> serde::Deserialize<'a>,
{
    let full_path = match path.canonicalize() {
        Ok(full_path) => Ok(full_path),
        Err(_) => Err(Error::Message(format!("path {:?} is wrong", path))),
    }?;
    let mut file = match std::fs::File::open(&full_path) {
        Ok(handle) => Ok(handle),
        Err(_) => Err(Error::Message(format!("file {:?} is not exist", full_path))),
    }?;
    let mut mats_string = String::new();
    match file.read_to_string(&mut mats_string) {
        Ok(_) => Ok(()),
        Err(_) => Err(Error::Message(format!(
            "read file {:?} to string failed",
            full_path
        ))),
    }?;
    let mats: Vec<Matrix<T, ROW, COL>> = match serde_json::from_str(&mats_string) {
        Ok(matrices) => Ok(matrices),
        Err(_) => Err(Error::Message(format!(
            "file {:?} contents wrong format",
            path
        ))),
    }?;
    match mats.get(index - 1) {
        Some(mat) => Ok(mat.to_owned()),
        None => Err(Error::Message(format!(
            "read index {} out of boundary",
            index
        ))),
    }
}

#[cfg(feature = "serde_mat")]
#[doc(cfg(feature = "serde_mat"))]
/// store matrices into json file
///
/// ```rust
/// # use rmatrix_ks::error::Result;
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::utils::serde::{read, write};
/// # fn main() -> Result<()> {
/// let mat1 = Matrix::<f64, 3, 3>::create(vec![1.0, 2.0, 3.0, 4.0, 8.0, 7.0, 5.0, 9.0, 6.0])?;
/// let mat2 = Matrix::<f64, 3, 3>::create(vec![1.0, 3.0, 2.0, 4.0, 8.0, 7.0, 5.0, 9.0, 6.0])?;
/// write(&[&mat1, &mat2], std::path::Path::new("data/test.json"))?;
/// # Ok(())
/// # }
/// ```
pub fn write<T, const ROW: usize, const COL: usize>(
    mats: &[&Matrix<T, ROW, COL>],
    path: &std::path::Path,
) -> Result<()>
where
    T: std::fmt::Debug + serde::Serialize,
{
    let full_path = match path.canonicalize() {
        Ok(full_path) => Ok(full_path),
        Err(_) => Err(Error::Message(format!("path {:?} is wrong", path))),
    }?;
    let mut file = match std::fs::File::create(&full_path) {
        Ok(handle) => Ok(handle),
        Err(_) => Err(Error::Message(format!(
            "create file {:?} failed",
            full_path
        ))),
    }?;
    let mats_string = match serde_json::to_string(&mats) {
        Ok(mats_string) => Ok(mats_string),
        Err(_) => Err(Error::Message(format!(
            "serialize {:?} to string failed",
            mats
        ))),
    }?;
    match file.write(mats_string.as_bytes()) {
        Ok(_) => Ok(()),
        Err(_) => Err(Error::Message(format!(
            "write to file {:?} failed",
            full_path
        ))),
    }
}
