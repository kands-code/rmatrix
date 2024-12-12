use crate::matrix::matrix::Matrix;

use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

pub fn to_file<N, P, const R: usize, const C: usize>(m: &Matrix<N, R, C>, path: P)
where
    N: std::fmt::Debug,
    P: AsRef<Path> + std::fmt::Debug,
{
    use std::io::Write;

    let file = File::create(&path).expect(&format!(
        "Error[matrix::serde::to_file]: create file {:?} failed",
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
        .expect("Error[matrix::serde::to_file]: write to file failed");
}

pub fn from_file<N, P, const R: usize, const C: usize>(path: P) -> Option<Matrix<N, R, C>>
where
    N: Clone + std::str::FromStr,
    P: AsRef<Path> + std::fmt::Debug,
{
    use std::io::Read;

    let file = File::open(&path).expect(&format!(
        "Error[matrix::serde::from_file]: failed to open file {:?}",
        path
    ));
    let mut reader = BufReader::new(file);
    let mut data = String::new();
    reader
        .read_to_string(&mut data)
        .expect("Error[matrix::serde::from_file]: read as string failed");
    let inner = data
        .split(",")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<N>().ok().expect(&format!(
                "Error[matrix::serde::from_file]: parse {} from str failed",
                s
            ))
        })
        .collect::<Vec<N>>();
    Matrix::of(&inner)
}

pub fn from_stdin<N, const R: usize, const C: usize>() -> Option<Matrix<N, R, C>>
where
    N: std::str::FromStr,
{
    let stdin = std::io::stdin();
    let mut buffer = String::new();
    let mut inner = Vec::with_capacity(R * C);
    while inner.len() < R * C {
        stdin
            .read_line(&mut buffer)
            .expect("Error[matrix::serde::from_stdin]: read from stdin failed");
        buffer
            .trim()
            .split(|c: char| c.is_whitespace() || c == ',')
            .filter(|s| !s.is_empty())
            .for_each(|s| {
                let e = s.parse::<N>().ok().expect(&format!(
                    "Error[matrix::serde::from_file]: parse {} from str failed",
                    s
                ));
                inner.push(e);
            });
        buffer.clear();
    }
    Some(Matrix { inner })
}
