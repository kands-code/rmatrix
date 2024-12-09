//! # Number Trait :: Fractional
//!
//! A type that implements this trait is a fractional number.

use crate::number::{instances::ratio::Rational, traits::number::Number};

/// Fractional
pub trait Fractional: Number
where
    Self: std::ops::Div<Output = Self>,
{
    /// 0.5 or 1/2
    fn half() -> Self;

    /// reciprocal of a fraction
    fn reciprocal(self) -> Self;

    /// convert a rational number to a fractional number
    fn from_rational(rational_number: Rational) -> Self;
}
