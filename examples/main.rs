//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{complex::induced_l2_matrix_norm, matrix::Matrix},
    number::instances::{complex::Complex, float::Float},
};

fn main() {
    let m = Matrix::<Complex<Float>, 3, 3>::of(
        &[
            (1.0, -2.0),
            (3.0, -4.0),
            (5.0, 6.0),
            (7.0, -8.0),
            (9.0, 10.0),
            (11.0, -12.0),
            (13.0, 14.0),
            (15.0, -16.0),
            (17.0, 18.0),
        ]
        .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
    )
    .unwrap();
    let n = induced_l2_matrix_norm(&m);
    assert_eq!(n, Float::of(40.2086));
}
