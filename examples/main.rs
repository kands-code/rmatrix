//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{extra::qr_decomposition_h, matrix::Matrix},
    number::instances::double::Double,
};

fn main() {
    let m = Matrix::<Double, 4, 3>::of(
        &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 4.0].map(Double::of),
    )
    .unwrap();
    let (q, r) = qr_decomposition_h(&m);
    let q_expect = Matrix::<Double, 4, 4>::of(
        &[
            -0.5,
            -0.5,
            1.0 / (2.0 * 13.0f64.sqrt()),
            -5.0 / (2.0 * 13.0f64.sqrt()),
            -0.5,
            -0.5,
            -1.0 / (2.0 * 13.0f64.sqrt()),
            5.0 / (2.0 * 13.0f64.sqrt()),
            -0.5,
            0.5,
            -5.0 / (2.0 * 13.0f64.sqrt()),
            -1.0 / (2.0 * 13.0f64.sqrt()),
            -0.5,
            0.5,
            5.0 / (2.0 * 13.0f64.sqrt()),
            1.0 / (2.0 * 13.0f64.sqrt()),
        ]
        .map(Double::of),
    )
    .unwrap();
    assert_eq!(q, q_expect);
    let r_expect = Matrix::<Double, 4, 3>::of(
        &[
            -2.0,
            -1.0,
            -2.0,
            0.0,
            -1.0,
            1.0,
            0.0,
            0.0,
            13.0f64.sqrt(),
            0.0,
            0.0,
            0.0,
        ]
        .map(Double::of),
    )
    .unwrap();
    assert_eq!(r, r_expect);
}
