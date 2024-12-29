//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{extra::eigen_system_qr, matrix::Matrix},
    number::instances::float::Float,
};

fn main() {
    let m = Matrix::<Float, 4, 4>::of(
        &[
            1.0, 5.0, 2.0, 0.0, 5.0, 2.0, 5.0, 2.0, 2.0, 5.0, 3.0, 5.0, 0.0, 2.0, 5.0, 4.0,
        ]
        .map(Float::of),
    )
    .unwrap();
    let (es, evs) = eigen_system_qr(&m, 1024);
    println!("{}", es);
    println!("{}", evs);
}
