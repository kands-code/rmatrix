#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{math::is_square_matrix, matrix::Matrix},
    number::instances::word8::Word8,
};

fn main() {
    let m1 = Matrix::<Word8, 2, 2>::of(&[Word8::of(1), Word8::of(2), Word8::of(2), Word8::of(1)])
        .unwrap();
    let m2 = Matrix::<Word8, 2, 1>::of(&[Word8::of(1), Word8::of(3)]).unwrap();
    assert!(is_square_matrix(&m1));
    assert!(!is_square_matrix(&m2));
}
