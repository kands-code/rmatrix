//! # Complex
//!
//! complex number
//!
//! just an example

use crate::error::Error;
use crate::error::Result;
use crate::number::Fractional;
use crate::number::Number;
use crate::number::One;
use crate::number::Zero;

#[cfg(feature = "serde_mat")]
use serde::{Deserialize, Serialize};

/// complex number
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde_mat", derive(Serialize, Deserialize))]
pub struct Complex<T: Number> {
    pub real: T,
    pub imag: T,
}

impl<T: Number> Complex<T> {
    /// create a complex number
    ///
    /// ```rust
    /// # use rmatrix_ks::cmplx;
    /// # use rmatrix_ks::complex::Complex;
    /// # fn main() {
    /// let _ = Complex::create(1i32, 2i32); // 1 + 2I
    /// // or can use cmplx!()
    /// let _ = cmplx!(3i32, 4i32); // 3 + 4I
    /// # }
    /// ```
    pub fn create(real: T, imag: T) -> Self {
        Complex { real, imag }
    }

    /// norm for complex number
    ///
    /// norm(a + bI) = sqrt(a^2 + b^2)
    ///
    /// ```rust
    /// # use rmatrix_ks::complex::Complex;
    /// # fn main() {
    /// let c = Complex::create(3.0f32, 4.0f32); // 3 + 4I
    /// assert_eq!(5.0f32, c.norm());
    /// # }
    /// ```
    pub fn norm(&self) -> T
    where
        T: Fractional,
    {
        (self.real.to_owned() * self.real.to_owned() + self.imag.to_owned() * self.imag.to_owned())
            .sqrt()
    }
}

#[macro_export]
/// equivalent to Complex::create()
macro_rules! cmplx {
    ($x:expr, $y:expr) => {
        Complex::create($x, $y)
    };
}

impl<T: Number> std::fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<{}, {}>", self.real, self.imag)
    }
}

impl<T: Number> std::ops::Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::create(-self.real, -self.imag)
    }
}

impl<T: Number> std::ops::Add for Complex<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::create(self.real + rhs.real, self.imag + rhs.imag)
    }
}

impl<T: Number> std::iter::Sum for Complex<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, e| acc + e)
    }
}

impl<T: Number> std::ops::Sub for Complex<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::create(self.real - rhs.real, self.imag - rhs.imag)
    }
}

impl<T: Number> std::ops::Mul for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::create(
            self.real.to_owned() * rhs.real.to_owned() - self.imag.to_owned() * rhs.imag.to_owned(),
            self.real * rhs.imag + self.imag * rhs.real,
        )
    }
}

impl<T: Number> Default for Complex<T> {
    fn default() -> Self {
        Self {
            real: T::zero(),
            imag: T::zero(),
        }
    }
}

impl<T: Number> Zero for Complex<T> {
    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.imag.is_zero()
    }
}

impl<T: Number> One for Complex<T> {
    fn one() -> Self {
        Self::create(T::one(), T::zero())
    }

    fn is_one(&self) -> bool {
        self.real.is_one() && self.imag.is_zero()
    }
}

/// complex number is a  number
impl<T: Number> Number for Complex<T> {
    /// absolute value for complex number
    ///
    /// not the norm of complex number
    fn abs(self) -> Self {
        Self::create(self.real.abs(), self.imag.abs())
    }

    /// conjugate for complex number
    ///
    /// conj(a + bI) = a - bI
    ///
    /// ```rust
    /// # use rmatrix_ks::complex::Complex;
    /// # use rmatrix_ks::cmplx;
    /// # use rmatrix_ks::number::Number;
    /// # fn main() {
    /// let c = Complex::create(3.0f32, 4.0f32); // 3 + 4I
    /// assert_eq!(cmplx!(3.0f32, -4.0f32), c.conjugate());
    /// # }
    /// ```
    fn conjugate(self) -> Self {
        Complex::create(self.real, -self.imag)
    }

    fn ndiv(self, rhs: Self) -> Result<Self> {
        if rhs.is_zero() {
            Err(Error::DividedByZero)
        } else {
            let under = rhs.real.to_owned() * rhs.real.to_owned()
                + rhs.imag.to_owned() * rhs.imag.to_owned();
            Ok(Complex::create(
                (self.real.to_owned() * rhs.real.to_owned()
                    + self.imag.to_owned() * rhs.imag.to_owned())
                .ndiv(under.to_owned())?,
                (self.imag * rhs.real - self.real * rhs.imag).ndiv(under)?,
            ))
        }
    }
}
