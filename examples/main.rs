//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]

use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};

fn main() {
    let m = Matrix::fills(2, 2, Word8::of(2));
    let m_expect = Matrix::of(2, 2, &[2, 2, 2, 2].map(Word8::of)).unwrap();
    assert_eq!(m, m_expect);
}
