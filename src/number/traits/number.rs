//! # traits::number
//!
//! Types that implement this trait can be considered as numbers.

use crate::number::{
    instances::integer::Integer,
    traits::{one::One, zero::Zero},
};

/// Concept of NUMBER.
///
/// Numbers should be `Sized`,
/// support addition and multiplication,
/// and allow for subtraction, having opposites and absolute values.
/// Additionally, numbers should be able to undergo equality checks.
///
/// However, not all numbers are ordered;
/// for example, complex numbers do not have an order,
/// and not all numbers support division, such as integers.
pub trait Number: One + Zero
where
    Self: std::marker::Sized
        + std::marker::Send
        + std::marker::Sync
        + std::clone::Clone
        + std::default::Default
        + std::fmt::Debug
        + std::fmt::Display
        + std::cmp::PartialEq
        + std::ops::Neg<Output = Self>
        + std::ops::Add<Output = Self>
        + std::ops::Sub<Output = Self>
        + std::ops::Mul<Output = Self>
        + std::str::FromStr<Err = ()>,
{
    /// Calculate the absolute value of a number.
    fn absolute_value(&self) -> Self;

    /// Calculate the sign of a number.
    fn sign_number(&self) -> Self;

    /// Convert an integer to another numeric type.
    fn from_integer(integer_number: Integer) -> Self;
}
