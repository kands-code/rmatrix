#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::{matrix::serde::from_stdin, number::instances::word8::Word8};

fn main() {
    let m = from_stdin::<Word8, 3, 3>();
    assert_eq!(m.dimension(), (3, 3));
}
