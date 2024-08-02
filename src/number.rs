//! # Number
//!
//! define the basic number typeclass

use crate::error::MatrixError;

/// concept of zero
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

/// concept of one
pub trait One {
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
/// impl Zero for f32 {
///     fn is_zero(&self) -> bool {
///         self.abs() < f32::EPSILON
///     }
/// }
///
/// impl One for f32 {
///     fn one() -> Self {
///         1.0f32
///     }
///
///     fn is_one(&self) -> bool {
///         (self - Self::one()).is_zero()
///     }
/// }
///
/// impl Number for f32 {
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
        + std::fmt::Display
        + std::iter::Sum
        + std::ops::Add<Output = Self>
        + std::ops::Sub<Output = Self>
        + std::ops::Mul<Output = Self>
        + std::ops::Neg<Output = Self>,
{
    /// absolute value
    fn abs(self) -> Self;

    /// normal division with zero test
    fn ndiv(self, rhs: Self) -> Result<Self, MatrixError>;
}

impl Zero for i32 {
    fn is_zero(&self) -> bool {
        self == &0i32
    }
}

impl One for i32 {
    fn one() -> Self {
        1i32
    }

    fn is_one(&self) -> bool {
        self == &1i32
    }
}

impl Number for i32 {
    fn abs(self) -> Self {
        i32::abs(self)
    }

    fn ndiv(self, rhs: Self) -> Result<i32, MatrixError> {
        if rhs.is_zero() {
            Err(MatrixError::DividedByZero)
        } else {
            Ok(self / rhs)
        }
    }
}

impl Zero for f32 {
    fn is_zero(&self) -> bool {
        self.abs() < f32::EPSILON
    }
}

impl One for f32 {
    fn one() -> Self {
        1.0f32
    }

    fn is_one(&self) -> bool {
        (self - Self::one()).is_zero()
    }
}

impl Number for f32 {
    fn abs(self) -> Self {
        f32::abs(self)
    }

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
        self.abs() < f64::EPSILON
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
    fn abs(self) -> Self {
        f64::abs(self)
    }

    fn ndiv(self, rhs: Self) -> Result<Self, MatrixError> {
        if rhs.is_zero() {
            Err(MatrixError::DividedByZero)
        } else {
            Ok(self / rhs)
        }
    }
}
