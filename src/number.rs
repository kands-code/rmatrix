//! # Number
//!
//! define the basic number typeclass

use crate::error::MatrixError;

pub trait Zero
where
    Self: Default,
{
    /// the ZERO of the number type
    fn zero() -> Self {
        Self::default()
    }

    /// check whether a ZERO
    fn is_zero(&self) -> bool;
}

pub trait One
where
    Self: Default,
{
    /// the ONE of the number type
    fn one() -> Self;

    /// check whether a ONE
    fn is_one(&self) -> bool;
}

/// number typeclass
///
/// # Example
///
/// ```rust,ignore
/// impl Zero for i8 {
///     fn is_zero(&self) -> bool {
///         self.to_owned() == 0i8
///     }
/// }
///
/// impl One for i8 {
///     fn one() -> Self {
///         1i8
///     }
///
///     fn is_one(&self) -> bool {
///         self.to_owned() == Self::one()
///     }
/// }
///
/// impl Number for i8 {
///     fn ndiv(self, rhs: Self) -> Result<Self, MatrixError> {
///         if rhs.is_zero() {
///             Err(MatrixError::DividedByZero)
///         } else {
///             Ok(self / rhs)
///         }
///     }
/// }
/// ```
pub trait Number: Zero + One
where
    Self: std::clone::Clone
        + std::cmp::PartialEq
        + std::default::Default
        + std::fmt::Debug
        + std::iter::Sum
        + std::ops::Add<Output = Self>
        + std::ops::Sub<Output = Self>
        + std::ops::Mul<Output = Self>
        + std::ops::Neg<Output = Self>
        + std::str::FromStr,
{
    /// normal division with zero test
    fn ndiv(self, rhs: Self) -> Result<Self, MatrixError>;
}

impl Zero for i8 {
    fn is_zero(&self) -> bool {
        self.to_owned() == 0i8
    }
}

impl One for i8 {
    fn one() -> Self {
        1i8
    }

    fn is_one(&self) -> bool {
        self.to_owned() == Self::one()
    }
}

impl Number for i8 {
    fn ndiv(self, rhs: Self) -> Result<Self, MatrixError> {
        if rhs.is_zero() {
            Err(MatrixError::DividedByZero)
        } else {
            Ok(self / rhs)
        }
    }
}

impl Zero for f64 {
    fn is_zero(&self) -> bool {
        self < &f64::EPSILON
    }
}

impl One for f64 {
    fn one() -> Self {
        1.0f64
    }

    fn is_one(&self) -> bool {
        (self - Self::one()).is_zero()
    }
}

impl Number for f64 {
    fn ndiv(self, rhs: Self) -> Result<Self, MatrixError> {
        if rhs.is_zero() {
            Err(MatrixError::DividedByZero)
        } else {
            Ok(self / rhs)
        }
    }
}
