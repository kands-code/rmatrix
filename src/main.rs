#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{
    matrix::{matrix::Matrix, utils::vertical_concat},
    number::instances::word8::Word8,
};

fn main() {
    let m = Matrix::<Word8, 2, 2>::of(&[1, 2, 3, 4].map(|e| Word8::of(e))).unwrap();
    let n = Matrix::<Word8, 2, 2>::of(&[5, 6, 7, 8].map(|e| Word8::of(e))).unwrap();
    let cat = Matrix::<Word8, 4, 2>::of(&[1, 2, 3, 4, 5, 6, 7, 8].map(|e| Word8::of(e))).unwrap();
    assert_eq!(vertical_concat(&m, &n), cat);
}
