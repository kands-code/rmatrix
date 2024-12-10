//! # Number Trait
//!
//! A type that implements this trait is a number.

use crate::number::{
    instances::integer::Integer,
    traits::{one::One, zero::Zero},
};

/// Number
///
/// A number is [Sized]
/// and can be added, subtracted, multiplied,
/// has its opposite, can have its absolute value calculated,
/// and can be compared for equality,
/// but it may not be comparable, like complex numbers,
/// and it may not be divisible, like integers.
pub trait Number: One + Zero
where
    Self: std::marker::Sized
        + std::clone::Clone
        + std::default::Default
        + std::fmt::Debug
        + std::fmt::Display
        + std::cmp::PartialEq
        + std::ops::Neg<Output = Self>
        + std::ops::Add<Output = Self>
        + std::ops::Sub<Output = Self>
        + std::ops::Mul<Output = Self>
        + std::str::FromStr<Err = String>,
{
    /// absolute value
    fn absolute_value(&self) -> Self;

    /// unit value corresponding to the signs
    ///
    /// 0, 1 or -1
    fn sign_number(&self) -> Self;

    /// convert integer number to other number
    fn from_integer(integer_number: Integer) -> Self;
}
