#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{math::row_reduce, matrix::Matrix},
    number::{instances::double::Double, traits::zero::Zero},
};

fn main() {
    let m = Matrix::<Double, 3, 3>::of(
        &[
            2.0, 1.0, -1.0, // r1
            -3.0, -1.0, 2.0, // r2
            -2.0, 1.0, 2.0, // r3
        ]
        .map(|e| Double::of(e)),
    )
    .unwrap();
    let (_, _, _, reduced) = row_reduce(&m);
    assert!((reduced
        - Matrix::<Double, 3, 3>::of(
            &[
                2.0, 1.0, -1.0, //r1
                0.0, 0.5, 0.5, // r2
                0.0, 0.0, -1.0 // r3
            ]
            .map(|e| Double::of(e))
        )
        .unwrap())
    .linear_iter()
    .all(|e| e.is_zero()));
}
