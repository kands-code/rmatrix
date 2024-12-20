//! # traits::real
//!
//! Types that implement this trait can be considered as real numbers.

use crate::number::{instances::ratio::Rational, traits::number::Number};

/// Concepts of Real.
///
/// Real numbers are not only numbers
/// but also possess a partial ordering relation.
pub trait Real: Number
where
    Self: std::cmp::PartialOrd,
{
    /// Convert a real number to a rational number.
    fn to_rational(self) -> Rational;
}
