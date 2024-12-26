//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{complex::induced_l2_matrix_norm, matrix::Matrix},
    number::instances::{complex::Complex, double::Double},
};

fn main() {
    let m = Matrix::<Complex<Double>, 2, 2>::of(
        &[(1.0, 2.0), (3.0, -1.0), (4.0, 5.0), (6.0, -3.0)]
            .map(|(r, i)| Complex::of(Double::of(r), Double::of(i))),
    )
    .unwrap();
    let l2_norm = induced_l2_matrix_norm(&m);
    assert_eq!(
        l2_norm,
        Double::of(((101.0 + 10085.0f64.sqrt()) / 2.0).sqrt())
    );
}
