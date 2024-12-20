#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::number::{
    instances::{double::Double, integer::Integer},
    traits::number::Number,
};

fn main() {
    let i2 = Integer::of_str("123456789101112131415161718192021222324252627").unwrap();
    let d2 = Double::from_integer(i2);
    assert_eq!(
        d2,
        Double::of(123456789101112130000000000000000000000000000.0)
    );
}
