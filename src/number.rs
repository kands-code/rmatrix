//! basic number trait
//!
//! all number types should implement this trait
//!
//! this trait defines the basic operations and properties of numbers

/// Number Trait
///
/// if x, y belong to (Num a), then:
///
/// - x can be added to y, the result belongs to (Num a);
/// - x can be subtracted from y, the result belongs to (Num a);
/// - x can be multiplied by y, the result belongs to (Num a);
/// - x can be divided by y if y is not *ZERO*, the result belongs to (Num a);
/// - x has the opposite number, the opposite number belongs to (Num a);
/// - x and y can be compared for equality;
/// - x can be obtained from a string;
/// - of course, x and y can be added as iteration values.
pub trait INum<'de>:
    std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::cmp::PartialEq
    + std::default::Default
    + std::clone::Clone
    + std::fmt::Debug
    + std::fmt::Display
    + std::str::FromStr
    + std::iter::Sum
    + std::convert::Into<f64>
    + serde::Serialize
    + serde::Deserialize<'de>
{
    /// get the number *ONE*
    fn one() -> Self;

    /// check if self is *ZERO*
    fn is_zero(&self) -> bool;

    /// convert from f64
    fn from_f64(value: f64) -> Self;
}

impl<'de> INum<'de> for i32 {
    fn one() -> Self {
        1_i32
    }

    fn is_zero(&self) -> bool {
        &0_i32 == self
    }

    fn from_f64(value: f64) -> Self {
        value as i32
    }
}

impl<'de> INum<'de> for f32 {
    fn one() -> Self {
        1.0_f32
    }

    fn is_zero(&self) -> bool {
        10.0_f32.powi(-5) >= self.abs()
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl<'de> INum<'de> for f64 {
    fn one() -> Self {
        1.0_f64
    }

    fn is_zero(&self) -> bool {
        10.0_f64.powi(-7) >= self.abs()
    }

    fn from_f64(value: f64) -> Self {
        value
    }
}
