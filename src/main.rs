#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{math::inverse, matrix::Matrix},
    number::instances::double::Double,
};

fn main() {
    let m = Matrix::<Double, 3, 3>::of(
        &[2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0].map(|e| Double::of(e)),
    )
    .unwrap();
    let inv = inverse(&m).unwrap();
    assert_eq!(inv * m, Matrix::<Double, 3, 3>::eyes());
}
