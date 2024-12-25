//! # examples::main
//!
//! A simple demonstration of using this library.

#![warn(missing_docs)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use rmatrix_ks::number::{
    instances::{float::Float, ratio::Rational},
    traits::{floating::Floating, real::Real},
};

fn main() {
    let m = Float::PI;
    let m_rat = m.to_rational();
    let rat_expect = Rational::of_str("13176795 % 4194304").unwrap();
    assert_eq!(m_rat, rat_expect);
}
