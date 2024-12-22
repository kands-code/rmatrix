#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

//! # examples::main
//!
//! A simple demonstration of using this library.

use rmatrix_ks::{
    matrix::{extra::qr_decomposition_h, matrix::Matrix},
    number::instances::double::Double,
};

fn main() {
    let m = Matrix::<Double, 2, 3>::of(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(Double::of)).unwrap();
    let (q, r) = qr_decomposition_h(&m);
    println!("{}", q);
    println!("{}", r);
}
