//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{math::determinant_l, matrix::Matrix},
    number::instances::int::Int,
};

fn main() {
    let m = Matrix::<Int, 4, 4>::of(&[1, 2, 3, 4, 1, 3, 4, 1, 1, 4, 1, 2, 1, 1, 2, 3].map(Int::of))
        .unwrap();
    assert_eq!(determinant_l(&m), Some(Int::of(16)));
}
