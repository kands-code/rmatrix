//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]

use rmatrix_ks::{
    matrix::{
        complex::{is_unitary_matrix, qr_decomposition_es},
        matrix::Matrix,
    },
    number::instances::{complex::Complex, float::Float},
};

fn main() {
    let m = Matrix::<Complex<Float>>::of(
        3,
        2,
        &[
            (1.0, 1.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 2.0),
            (5.0, 0.0),
            (6.0, 0.0),
        ]
        .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
    )
    .unwrap();
    let (q, r) = qr_decomposition_es(&m);
    let q_expect = Matrix::<Complex<Float>>::of(
        3,
        2,
        &[
            (1.0 / 6.0, 1.0 / 6.0),
            (4.0 / 117.0f32.sqrt(), -2.0 / 13.0f32.sqrt()),
            (0.5, 0.0),
            (1.0 / 52.0f32.sqrt(), 5.0 / 52.0f32.sqrt()),
            (5.0 / 6.0, 0.0),
            (-1.0 / 468.0f32.sqrt(), -5.0 / 468.0f32.sqrt()),
        ]
        .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
    )
    .unwrap();
    assert!(is_unitary_matrix(&q));
    println!("{}", q);
    assert_eq!(q, q_expect);
    let r_expect = Matrix::<Complex<Float>>::of(
        2,
        2,
        &[
            (6.0, 0.0),
            (22.0 / 3.0, 2.0 / 3.0),
            (0.0, 0.0),
            (52.0f32.sqrt() / 3.0, 0.0),
        ]
        .map(|(r, i)| Complex::of(Float::of(r), Float::of(i))),
    )
    .unwrap();
    assert_eq!(r, r_expect);
}
