#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::vector::{root_mean_square, VectorC},
    number::{
        instances::float::Float,
        traits::{floating::Floating, zero::Zero},
    },
};

fn main() {
    let v: VectorC<Float, 3> =
        VectorC::of(&[Float::of(1.0), Float::of(2.0), Float::of(3.0)]).unwrap();
    assert!((root_mean_square(&v) - Float::of(14.0 / 3.0).square_root()).is_zero());
}
