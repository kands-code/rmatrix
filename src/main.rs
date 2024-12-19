#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::number::{
    instances::{float::Float, int::Int},
    traits::zero::Zero,
    utils::integral_power,
};

fn main() {
    let m = Int::of(-2);
    let n = Float::of(2.0);
    assert!((integral_power(n, m).is_some_and(|e| (e - Float::of(0.25)).is_zero())));
}
