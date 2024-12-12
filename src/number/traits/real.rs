//! # Number Trait :: Real
//!
//! A type that implements this trait is a real number.

use crate::number::{instances::ratio::Rational, traits::number::Number};

/// Real
pub trait Real: Number
where
    Self: std::cmp::PartialOrd,
{
    /// convert a real number to a rational number
    fn to_rational(self) -> Rational;
}
