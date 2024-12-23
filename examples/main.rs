//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{extra::eigen_system_qr, matrix::Matrix, vector::VectorC},
    number::instances::float::Float,
};

fn main() {
    let m =
        Matrix::<Float, 3, 3>::of(&[1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0].map(Float::of))
            .unwrap();
    let (e, ev) = eigen_system_qr(&m);
    assert_eq!(e.len(), ev.len());
    // This matrix has three eigenvalues and corresponding eigenvectors.
    assert_eq!(e.len(), 3);
    assert_eq!(e, [1.0, 4.0, 6.0].map(Float::of).to_vec());
    assert_eq!(
        ev[0],
        VectorC::<Float, 3>::of(&[1.0, 0.0, 0.0].map(Float::of)).unwrap()
    );
    assert_eq!(
        ev[1],
        VectorC::<Float, 3>::of(&[2.0 / 3.0, 1.0, 0.0].map(Float::of)).unwrap()
    );
    assert_eq!(
        ev[2],
        VectorC::<Float, 3>::of(&[1.6, 2.5, 1.0].map(Float::of)).unwrap()
    );
}
