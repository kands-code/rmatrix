//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]

use rmatrix_ks::{
    matrix::{
        matrix::Matrix,
        vector::{column_vector, is_column_vector},
    },
    number::instances::word8::Word8,
};

fn main() {
    let v = column_vector(3, &[1, 0, 0].map(Word8::of)).unwrap();
    assert!(is_column_vector(&v));
    let m = Matrix::of(2, 2, &[1, 0, 2, 5].map(Word8::of)).unwrap();
    assert!(!is_column_vector(&m));
}
