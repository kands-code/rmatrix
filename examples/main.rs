//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{extra::singular_value_decomposition, matrix::Matrix, utils::transpose},
    number::instances::float::Float,
};

fn main() {
    let m = Matrix::<Float, 3, 2>::of(
        &[2.0 / 3.0, 0.0, 5.0 / 6.0, 0.5, 1.0 / 3.0, 1.0].map(Float::of),
    )
    .unwrap();
    let (u, s, v) = singular_value_decomposition(&m);
    println!("{}", u);
    println!("{}", s);
    println!("{}", v);
    println!("{}", u * s * transpose(&v));
}
