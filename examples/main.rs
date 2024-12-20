#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::number::{instances::double::Double, traits::realfloat::RealFloat};

fn main() {
    let rfp = Double::of(3.14);
    let (s, e) = rfp.decode_float();
    println!("{}, {}", s, e);
}
