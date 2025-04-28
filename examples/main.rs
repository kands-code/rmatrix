//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{Matrix, math::row_eliminate},
    number::instances::float::Float,
};

fn main() {
    let m = Matrix::<Float, 3, 3>::of(&[2.0, 1.0, 3.0, 3.0, 1.0, 4.0, 5.0, 7.0, 12.0].map(Float::of)).unwrap();
    let r = row_eliminate(&m);
    println!("{r}");
}
