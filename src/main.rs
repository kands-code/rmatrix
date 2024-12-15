#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::vector::{angle_between, VectorC},
    number::{instances::double::Double, traits::zero::Zero},
};

fn main() {
    let v1: VectorC<Double, 3> =
        VectorC::of(&[Double::of(1.0), Double::of(5.0), Double::of(4.0)]).unwrap();
    let v2: VectorC<Double, 3> =
        VectorC::of(&[Double::of(8.0), Double::of(-4.0), Double::of(3.0)]).unwrap();
    let angle = angle_between(&v1, &v2);
    assert!(angle.is_some());
    assert!((angle.unwrap() - Double::of(core::f64::consts::PI / 2.0)).is_zero());

    // angle between a vector and itself is zero
    let self_angle = angle_between(&v1, &v1);
    assert!(self_angle.is_some());
    assert!(self_angle.unwrap().is_zero());

    // zero-vector has no angle with any other vector
    let v3 = VectorC::of(&[Double::zero(), Double::zero(), Double::zero()]).unwrap();
    assert!(angle_between(&v1, &v3).is_none());
}
