#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::vector::{maximum_norm, VectorC},
    number::instances::int8::Int8,
};

fn main() {
    let v = VectorC::<Int8, 3>::of(&[Int8::of(2), Int8::of(-5), Int8::of(3)]).unwrap();
    assert_eq!(maximum_norm(&v), Int8::of(5))
}
