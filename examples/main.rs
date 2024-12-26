//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{complex::givens_rotation_matrix, matrix::Matrix},
    number::instances::{complex::Complex, float::Float},
};

fn main() {
    let m = Matrix::<Complex<Float>, 2, 2>::of(
        &[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
            .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
    )
    .unwrap();
    let g = givens_rotation_matrix(&m, (2, 1), (1, 1));
    let g_expect = Matrix::<Complex<Float>, 2, 2>::of(
        &[
            (1.0 / 10.0f32.sqrt(), 0.0),
            (3.0 / 10.0f32.sqrt(), 0.0),
            (-3.0 / 10.0f32.sqrt(), 0.0),
            (1.0 / 10.0f32.sqrt(), 0.0),
        ]
        .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
    )
    .unwrap();
    assert_eq!(g, g_expect);
}
