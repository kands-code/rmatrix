//! # matrix::serde
//!
//! Serialize and deserialize the matrix,
//! including obtaining the matrix from standard input.

use crate::matrix::matrix::Matrix;

use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

/// Write the matrix to a file.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, serde::to_file},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let path = "data/test.txt";
///     let m = Matrix::<Complex<Float>, 16, 16>::rand(
///         Complex::of(Float::of(-2.0), Float::of(-2.0)),
///         Complex::of(Float::of(2.0), Float::of(2.0)),
///     );
///     to_file(&m, path);
/// }
/// ```
pub fn to_file<N, P, const R: usize, const C: usize>(m: &Matrix<N, R, C>, path: P)
where
    N: std::fmt::Debug,
    P: AsRef<Path> + std::fmt::Debug,
{
    use std::io::Write;

    let file = File::create(&path).expect(&format!(
        "Error[matrix::serde::to_file]: Failed to create file ({:?}).",
        path
    ));
    let mut writer = BufWriter::new(file);
    let data = m
        .linear_iter()
        .map(|e| format!("{:?}", e))
        .collect::<Vec<_>>()
        .join(",");
    writer
        .write_all(data.as_bytes())
        .expect("Error[matrix::serde::to_file]: Failed to write the matrix to the file.");
}

/// Read the matrix from a file.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, serde::from_file},
///     number::instances::{complex::Complex, float::Float},
/// };
///
/// fn main() {
///     let path = "data/test.txt";
///     let m: Option<Matrix<Complex<Float>, 16, 16>> = from_file(path);
///     assert!(m.is_some());
/// }
/// ```
pub fn from_file<N, P, const R: usize, const C: usize>(path: P) -> Option<Matrix<N, R, C>>
where
    N: Clone + std::str::FromStr,
    P: AsRef<Path> + std::fmt::Debug,
{
    use std::io::Read;

    let file = File::open(&path).expect(&format!(
        "Error[matrix::serde::from_file]: Failed to create file ({:?}).",
        path
    ));
    let mut reader = BufReader::new(file);
    let mut data = String::new();
    reader.read_to_string(&mut data).expect(
        "Error[matrix::serde::from_file]: Failed to read the file content into the string.",
    );
    let inner = data
        .split(",")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<N>().ok().expect(&format!(
                "Error[matrix::serde::from_file]: Failed to parse ({}) from the string.",
                s
            ))
        })
        .collect::<Vec<N>>();
    Matrix::of(&inner)
}

/// Read the matrix from user input.
///
/// # Examples
///
/// ```rust,no_run
/// use rmatrix_ks::{matrix::serde::from_stdin, number::instances::word8::Word8};
///
/// fn main() {
///     let m = from_stdin::<Word8, 3, 3>();
///     assert_eq!(m.shape(), (3, 3));
/// }
/// ```
pub fn from_stdin<N, const R: usize, const C: usize>() -> Matrix<N, R, C>
where
    N: std::str::FromStr,
{
    let stdin = std::io::stdin();
    let mut buffer = String::new();
    let mut inner = Vec::with_capacity(R * C);
    println!("Please enter the matrix elements ({}, {}):", R, C);
    while inner.len() < R * C {
        stdin
            .read_line(&mut buffer)
            .expect("Error[matrix::serde::from_stdin]: Failed to read data from 'stdin'.");
        buffer
            .trim()
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .for_each(|s| {
                let e = s.parse::<N>().ok().expect(&format!(
                    "Error[matrix::serde::from_file]: Failed to parse ({}) from the string.",
                    s
                ));
                inner.push(e);
            });
        // Prevent duplicate reads.
        buffer.clear();
    }
    Matrix { inner }
}
