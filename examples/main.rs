//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{extra::linear_solve, matrix::Matrix},
    number::instances::float::Float,
};

fn main() {
    // x
    let m = Matrix::<Float, 10, 2>::vandermonde(
        &[
            208.0, 152.0, 113.0, 227.0, 137.0, 238.0, 178.0, 104.0, 191.0, 130.0,
        ]
        .map(Float::of),
    )
    .unwrap();
    // y
    let b = Matrix::<Float, 10, 1>::of(
        &[21.6, 15.5, 10.4, 31.0, 13.0, 32.4, 19.0, 10.4, 19.0, 11.8].map(Float::of),
    )
    .unwrap();
    // y = sol[0] + sol[1] x
    let sol = linear_solve(&m, &b);
    let sol_expect = Matrix::<Float, 2, 1>::of(&[-8.6451, 0.1612].map(Float::of)).unwrap();
    assert_eq!(sol, sol_expect);
}
