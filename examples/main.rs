//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]

use rmatrix_ks::{
    matrix::{math::determinant_e, matrix::Matrix},
    number::instances::int8::Int8,
};

fn main() {
    let m = Matrix::of(
        4,
        4,
        &[1, 2, 3, 4, 1, 3, 4, 1, 1, 4, 1, 2, 1, 1, 2, 3].map(Int8::of),
    )
    .unwrap();
    assert_eq!(determinant_e(&m), Int8::of(16));
}
