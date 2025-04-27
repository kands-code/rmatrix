//! # matrix::serde
//!
//! Serialize and deserialize the matrix,
//! including obtaining the matrix from standard input.

use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use crate::matrix::Matrix;

/// Write the matrix to a file.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, serde::to_file},
///     number::instances::float::Float,
/// };
///
/// let path = "data/random.txt";
/// let m = Matrix::<Float>::rand(16, 16, Float::of(-2.0), Float::of(2.0));
/// to_file(&m, path);
/// ```
pub fn to_file<N, P>(m: &Matrix<N>, path: P)
where
    N: std::fmt::Debug,
    P: AsRef<Path> + std::fmt::Debug,
{
    use std::io::Write;

    let file =
        File::create(&path).unwrap_or_else(|_| panic!("Error[matrix::serde::to_file]: Failed to create file ({path:?})."));
    let mut writer = BufWriter::new(file);
    let data = m
        .linear_iter()
        .map(|e| format!("{e:?}"))
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
///     matrix::{Matrix, serde::from_file},
///     number::instances::float::Float,
/// };
///
/// let path = "data/test.txt";
/// let m: Option<Matrix<Float>> = from_file(16, 16, path);
/// assert!(m.is_some());
/// ```
pub fn from_file<N, P>(row: usize, column: usize, path: P) -> Option<Matrix<N>>
where
    N: Clone + std::str::FromStr,
    P: AsRef<Path> + std::fmt::Debug,
{
    use std::io::Read;

    let file =
        File::open(&path).unwrap_or_else(|_| panic!("Error[matrix::serde::from_file]: Failed to create file ({path:?})."));
    let mut reader = BufReader::new(file);
    let mut data = String::new();
    reader
        .read_to_string(&mut data)
        .expect("Error[matrix::serde::from_file]: Failed to read the file content into the string.");
    let inner = data
        .split(",")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<N>()
                .ok()
                .unwrap_or_else(|| panic!("Error[matrix::serde::from_file]: Failed to parse ({s}) from the string."))
        })
        .collect::<Vec<N>>();
    Matrix::of(row, column, &inner)
}

/// Read the matrix from user input.
///
/// # Examples
///
/// ```rust,no_run
/// use rmatrix_ks::{matrix::serde::from_stdin, number::instances::word8::Word8};
///
/// let m = from_stdin::<Word8>();
/// assert_eq!(m.shape(), (3, 3));
/// ```
pub fn from_stdin<N>() -> Matrix<N>
where
    N: std::str::FromStr,
{
    let stdin = std::io::stdin();
    let mut buffer = String::new();
    println!("Please eneter the matrix shape (two integers separated by a space or a comma):");
    stdin
        .read_line(&mut buffer)
        .expect("Error[matrix::serde::from_stdin]: Failed to read shape from 'stdin'.");
    let shape = buffer
        .trim()
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .take(2)
        .map(|s| {
            s.parse::<usize>()
                .unwrap_or_else(|_| panic!("Error[matrix::serde::from_file]: Failed to parse ({s}) from the string."))
        })
        .collect::<Vec<usize>>();
    let row = shape[0];
    let column = shape[1];
    let mut inner = Vec::with_capacity(row * column);
    println!("Please enter the matrix elements ({row}, {column}):");
    while inner.len() < row * column {
        stdin
            .read_line(&mut buffer)
            .expect("Error[matrix::serde::from_stdin]: Failed to read data from 'stdin'.");
        buffer
            .trim()
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .for_each(|s| {
                let e = s
                    .parse::<N>()
                    .ok()
                    .unwrap_or_else(|| panic!("Error[matrix::serde::from_file]: Failed to parse ({s}) from the string."));
                inner.push(e);
            });
        // Prevent duplicate reads.
        buffer.clear();
    }
    Matrix { inner, row, column }
}
