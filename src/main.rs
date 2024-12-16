#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{math::is_lower_triangular_matrix, matrix::Matrix},
    number::instances::word8::Word8,
};

fn main() {
    let m1: Matrix<Word8, 2, 2> =
        Matrix::of(&[Word8::of(1), Word8::of(1), Word8::of(0), Word8::of(1)]).unwrap();
    assert!(!is_lower_triangular_matrix(&m1));

    let m2: Matrix<Word8, 2, 2> =
        Matrix::of(&[Word8::of(1), Word8::of(0), Word8::of(1), Word8::of(1)]).unwrap();
    assert!(is_lower_triangular_matrix(&m2));
}
