//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{extra::linear_solve_w, matrix::Matrix},
    number::instances::float::Float,
};

fn main() {
    // M
    let m = Matrix::<Float, 2, 3>::of(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(Float::of)).unwrap();
    // b
    let b = Matrix::<Float, 2, 1>::of(&[7.0, 8.0].map(Float::of)).unwrap();
    let sol = linear_solve_w(&m, &b);
    // Will return one of the possible solutions.
    let sol_expect = Matrix::<Float, 3, 1>::of(&[-3.0556, 0.1111, 3.2778].map(Float::of)).unwrap();
    assert_eq!(sol, sol_expect);
}
