//! # Number
//!
//! define the basic number typeclass

use crate::error::Error;
use crate::error::Result;

/// concept of equality
pub trait Equal {
    /// check whether two element are EQUAL
    fn equal(&self, rhs: &Self) -> bool;
}

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
/// impl Equal for f32 {
///     fn equal(&self, rhs: &Self) -> bool {
///         (self - rhs).is_zero()
///     }
/// }
///
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
///     fn abs(self) -> Self {
///         f32::abs(self)
///     }
///
///     fn conjugate(self) -> Self {
///         self
///     }
///
///     fn ndiv(self, rhs: Self) -> Result<Self> {
///         if rhs.is_zero() {
///             Err(Error::DividedByZero)
///         } else {
///             Ok(self / rhs)
///         }
///     }
/// }
/// ```
pub trait Number: Equal + Zero + One
where
    Self: std::clone::Clone
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

    /// conjugate
    fn conjugate(self) -> Self;

    /// normal division with zero test
    fn ndiv(self, rhs: Self) -> Result<Self>;
}

impl Equal for i32 {
    fn equal(&self, rhs: &Self) -> bool {
        self == rhs
    }
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

    fn conjugate(self) -> Self {
        self
    }

    fn ndiv(self, rhs: Self) -> Result<Self> {
        if rhs.is_zero() {
            Err(Error::DividedByZero)
        } else {
            Ok(self / rhs)
        }
    }
}

impl Equal for f32 {
    fn equal(&self, rhs: &Self) -> bool {
        (self - rhs).is_zero()
    }
}

impl Zero for f32 {
    /// if a f32 number smaller than sqrt(eps),
    /// then we can say it is zero
    fn is_zero(&self) -> bool {
        self.abs() < f32::EPSILON.sqrt()
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

    fn conjugate(self) -> Self {
        self
    }

    fn ndiv(self, rhs: Self) -> Result<Self> {
        if rhs.is_zero() {
            Err(Error::DividedByZero)
        } else {
            Ok(self / rhs)
        }
    }
}

/// concept for integeral number
pub trait Integeral
where
    Self: Number + std::cmp::PartialOrd + std::ops::Rem<Output = Self>,
{
    /// greatest common divisor
    fn gcd(&self, rhs: &Self) -> Self {
        if rhs.to_owned().equal(&Self::zero()) {
            // gcd is non-negative
            self.to_owned().abs()
        } else {
            rhs.gcd(&(self.to_owned().rem(rhs.to_owned())))
        }
    }

    /// least common multiple
    fn lcm(&self, rhs: &Self) -> Self {
        // |a * b| = gcd(a, b) * lcm(a, b)
        match (self.to_owned() * rhs.to_owned()).abs().ndiv(self.gcd(rhs)) {
            Ok(division) => division,
            Err(_) => -Self::one(),
        }
    }
}

impl Integeral for i32 {}

/// concept for floating point number
pub trait Fractional
where
    Self: Number,
{
    /// square root for fractional
    fn nsqrt(self) -> Result<Self>;

    /// convert size to fractional for computation
    fn from_usize(size: usize) -> Self;
}

impl Fractional for f32 {
    fn nsqrt(self) -> Result<Self> {
        Ok(self.sqrt())
    }

    fn from_usize(size: usize) -> Self {
        size as f32
    }
}
