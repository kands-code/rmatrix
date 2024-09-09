//! # Number
//!
//! define the basic number typeclass

use crate::error::IError;
use crate::error::IResult;

/// concept of equality
pub trait IEqual {
    /// check whether two element are EQUAL
    fn equal(&self, rhs: &Self) -> bool;
}

/// concept of zero
pub trait IZero
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
pub trait IOne {
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
/// impl IEqual for f32 {
///     fn equal(&self, rhs: &Self) -> bool {
///         (self - rhs).is_zero()
///     }
/// }
/// impl IZero for f32 {
///     fn is_zero(&self) -> bool {
///         self.abs() < f32::EPSILON
///     }
/// }
/// impl IOne for f32 {
///     fn one() -> Self {
///         1.0f32
///     }
///     fn is_one(&self) -> bool {
///         (self - Self::one()).is_zero()
///     }
/// }
/// impl INumber for f32 {
///     fn abs(self) -> Self {
///         f32::abs(self)
///     }
///     fn conjugate(self) -> Self {
///         self
///     }
///     fn ndiv(self, rhs: Self) -> Result<Self> {
///         if rhs.is_zero() {
///             Err(Error::DividedByZero)
///         } else {
///             Ok(self / rhs)
///         }
///     }
/// }
/// ```
pub trait INumber: IEqual + IZero + IOne
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
    fn ndiv(self, rhs: Self) -> IResult<Self>;
}

/// implementation of equal trait for i32
impl IEqual for i32 {
    fn equal(&self, rhs: &Self) -> bool {
        self == rhs
    }
}
/// implementation of zero trait for i32
impl IZero for i32 {
    fn is_zero(&self) -> bool {
        self == &0i32
    }
}
/// implementation of one trait for i32
impl IOne for i32 {
    fn one() -> Self {
        1i32
    }

    fn is_one(&self) -> bool {
        self == &1i32
    }
}
/// implementation of number trait for i32
impl INumber for i32 {
    fn abs(self) -> Self {
        i32::abs(self)
    }

    fn conjugate(self) -> Self {
        self
    }

    fn ndiv(self, rhs: Self) -> IResult<Self> {
        if rhs.is_zero() {
            Err(IError::DividedByZero)
        } else {
            Ok(self / rhs)
        }
    }
}

/// implementation of equal trait for f32
impl IEqual for f32 {
    fn equal(&self, rhs: &Self) -> bool {
        (self - rhs).is_zero()
    }
}
/// implementation of zero trait for f32
impl IZero for f32 {
    /// if a f32 number smaller than sqrt(eps),
    /// then we can say it is zero
    fn is_zero(&self) -> bool {
        self.abs() < f32::EPSILON.sqrt()
    }
}
/// implementation of one trait for f32
impl IOne for f32 {
    fn one() -> Self {
        1.0f32
    }
    fn is_one(&self) -> bool {
        (self - Self::one()).is_zero()
    }
}
/// implementation of number trait for f32
impl INumber for f32 {
    fn abs(self) -> Self {
        f32::abs(self)
    }

    fn conjugate(self) -> Self {
        self
    }

    fn ndiv(self, rhs: Self) -> IResult<Self> {
        if rhs.is_zero() {
            Err(IError::DividedByZero)
        } else {
            Ok(self / rhs)
        }
    }
}

/// concept for integeral number
pub trait IIntegeral: INumber
where
    Self: std::cmp::PartialOrd + std::ops::Rem<Output = Self>,
{
    /// greatest common divisor
    fn gcd(&self, rhs: &Self) -> Self {
        if rhs.clone().equal(&Self::zero()) {
            // gcd is non-negative
            self.clone().abs()
        } else {
            rhs.gcd(&(self.clone().rem(rhs.clone())))
        }
    }

    /// least common multiple
    fn lcm(&self, rhs: &Self) -> Self {
        // |a * b| = gcd(a, b) * lcm(a, b)
        match (self.clone() * rhs.clone()).abs().ndiv(self.gcd(rhs)) {
            Ok(division) => division,
            Err(_) => -Self::one(),
        }
    }
}

impl IIntegeral for i32 {}

/// concept for floating point number
pub trait IFractional: INumber {
    /// square root for fractional
    fn nsqrt(self) -> IResult<Self>;

    /// convert size to fractional for computation
    fn from_usize(size: usize) -> Self;
}

impl IFractional for f32 {
    fn nsqrt(self) -> IResult<Self> {
        Ok(self.sqrt())
    }

    fn from_usize(size: usize) -> Self {
        size as f32
    }
}
