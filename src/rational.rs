//! # Relation
//!
//! relation number
//!
//! ## Warning
//!
//! very slow, and may overflow

use crate::error::Error;
use crate::error::Result;
use crate::number::Integeral;
use crate::number::Number;
use crate::number::One;
use crate::number::Zero;

#[cfg(feature = "serde_mat")]
use serde::{Deserialize, Serialize};

/// rational number
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde_mat", derive(Serialize, Deserialize))]
pub struct Rational<T: Integeral> {
    pub numerator: T,
    pub denominator: T,
}

impl<T: Integeral> Rational<T> {
    /// create a rational number
    ///
    /// ```rust
    /// # use rmatrix_ks::rat;
    /// # use rmatrix_ks::rational::Rational;
    /// # fn main() {
    /// let _ = Rational::create(1i32, 2i32); // 1/2
    /// // or can use rat!()
    /// let _ = rat!(3i32, 4i32); // 3/4
    /// # }
    /// ```
    pub fn create(numerator: T, denominator: T) -> Self {
        Rational {
            numerator,
            denominator,
        }
    }

    /// refine the format of the rational number
    ///
    /// by default, rational number will do not simplify,
    /// if an overflow error occurs, try this function to do simplify
    ///
    /// ```rust
    /// # use rmatrix_ks::error::Error;
    /// # use rmatrix_ks::error::Result;
    /// # use rmatrix_ks::rat;
    /// # use rmatrix_ks::rational::Rational;
    /// # fn main() -> Result<()> {
    /// let mut r1 = Rational::create(2i32, 4i32); // 2/4
    /// r1.refine()?; // 1/2
    /// # Ok(())
    /// # }
    /// ```
    pub fn refine(&self) -> Result<Self> {
        let gcd = self.numerator.gcd(&self.denominator);
        let mut numerator = self.numerator.to_owned().ndiv(gcd.to_owned())?;
        let mut denominator = self.denominator.to_owned().ndiv(gcd)?;
        // refine negative
        if (numerator < T::zero() && denominator < T::zero())
            || (numerator > T::zero() && denominator < T::zero())
        {
            numerator = -numerator;
            denominator = -denominator;
        } else if self.numerator.is_zero() {
            numerator = T::zero();
            denominator = T::one();
        }
        Ok(Self::create(numerator, denominator))
    }
}

#[macro_export]
/// equivalent to Rational::create()
macro_rules! rat {
    ($x:expr, $y:expr) => {
        Rational::create($x, $y)
    };
}

impl<T: Integeral> std::fmt::Display for Rational<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // 0/0 means nan
        let refined = match self.refine() {
            Ok(v) => v,
            Err(_) => Rational::create(T::zero(), T::zero()),
        };
        write!(f, "{}/{}", refined.numerator, refined.denominator)
    }
}

impl<T: Integeral> std::ops::Neg for Rational<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::create(-self.numerator, self.denominator)
    }
}

impl<T: Integeral> std::ops::Add for Rational<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let numerator = self.numerator * rhs.denominator.to_owned()
            + rhs.numerator * self.denominator.to_owned();
        let denominator = self.denominator * rhs.denominator;
        Self::create(numerator, denominator)
    }
}

impl<T: Integeral> std::iter::Sum for Rational<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, e| acc + e)
    }
}

impl<T: Integeral> std::ops::Sub for Rational<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let numerator = self.numerator * rhs.denominator.to_owned()
            - rhs.numerator * self.denominator.to_owned();
        let denominator = self.denominator * rhs.denominator;
        Self::create(numerator, denominator)
    }
}

impl<T: Integeral> std::ops::Mul for Rational<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let numerator = self.numerator * rhs.numerator;
        let denominator = self.denominator * rhs.denominator;
        Self::create(numerator, denominator)
    }
}

impl<T: Integeral> Default for Rational<T> {
    fn default() -> Self {
        Self {
            numerator: T::zero(),
            denominator: T::one(),
        }
    }
}

impl<T: Integeral> Zero for Rational<T> {
    fn is_zero(&self) -> bool {
        self.numerator.is_zero() && !self.denominator.is_zero()
    }
}

impl<T: Integeral> One for Rational<T> {
    fn one() -> Self {
        Self::create(T::one(), T::one())
    }

    fn is_one(&self) -> bool {
        self.numerator.gcd(&self.denominator).is_one()
    }
}

/// rational number is a  number
impl<T: Integeral> Number for Rational<T> {
    fn abs(self) -> Self {
        Self::create(self.numerator.abs(), self.denominator.abs())
    }

    fn conjugate(self) -> Self {
        rat!(self.numerator.conjugate(), self.denominator.conjugate())
    }

    /// (a / b) / (c / d) = (a * d) / (b * c)
    fn ndiv(self, rhs: Self) -> Result<Self> {
        if rhs.is_zero() {
            Err(Error::DividedByZero)
        } else {
            Ok(self * Self::create(rhs.denominator, rhs.numerator))
        }
    }
}
