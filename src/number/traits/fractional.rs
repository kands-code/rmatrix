//! # traits::fractional
//!
//! Types that implement this trait can be considered as fractional numbers.

use crate::number::{instances::ratio::Rational, traits::number::Number};

/// Concepts of Fractional.
///
/// Fractional numbers, in addition to being numbers,
/// must also be able to perform division operations.
pub trait Fractional: Number
where
    Self: std::ops::Div<Output = Self>,
{
    /// The concept of `1/2`, or in other words, half of ONE.
    fn half() -> Self;

    /// The reciprocal of a fractional number.
    fn reciprocal(self) -> Self;

    /// Converting a rational number into a fractional number.
    fn from_rational(rational_number: Rational) -> Self;
}
