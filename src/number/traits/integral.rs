//! # traits::integral
//!
//! Types that implement this trait can be considered as integral numbers.

use crate::number::{instances::integer::Integer, traits::real::Real};

/// Concepts of Integral.
///
/// Integral numbers should be real numbers
/// and have a total order relationship.
pub trait Integral: Real
where
    Self: std::cmp::Ord,
{
    /// Calculating the quotient of two integers, rounding the result towards zero.
    fn quotient(self, rhs: Self) -> Self { self.quot_rem(rhs).0 }

    /// Calculating the remainder corresponding to the quotient rounded towards zero.
    fn remainder(self, rhs: Self) -> Self { self.quot_rem(rhs).1 }

    /// Calculating the quotient of two integers, rounding the result towards negative infinity.
    fn division(self, rhs: Self) -> Self { self.div_mod(rhs).0 }

    /// Calculating the modulus of two integers.
    fn modulus(self, rhs: Self) -> Self { self.div_mod(rhs).1 }

    /// Calculating the quotient and remainder of two integers,
    /// where the quotient is rounded towards zero.
    ///
    /// For example, for the `Integer`:
    ///
    /// ```rust
    /// use rmatrix_ks::number::{instances::integer::Integer, traits::integral::Integral};
    ///
    /// let i1 = Integer::of_str("8").unwrap();
    /// let i2 = Integer::of_str("-3").unwrap();
    /// let (quot, rem) = i1.clone().quot_rem(i2.clone());
    /// assert_eq!(quot, Integer::of_str("-2").unwrap());
    /// assert_eq!(rem, Integer::of_str("2").unwrap());
    /// assert_eq!(quot * i2 + rem, i1);
    /// ```
    fn quot_rem(self, rhs: Self) -> (Self, Self);

    /// Calculating the division and modulus of two integers,
    /// where the division result is rounded towards negative infinity.
    ///
    /// For example, for the `Integer`:
    ///
    /// ```rust
    /// use rmatrix_ks::number::{instances::integer::Integer, traits::integral::Integral};
    ///
    /// let i1 = Integer::of_str("8").unwrap();
    /// let i2 = Integer::of_str("-3").unwrap();
    /// let (div, m) = i1.clone().div_mod(i2.clone());
    /// assert_eq!(div, Integer::of_str("-3").unwrap());
    /// assert_eq!(m, Integer::of_str("-1").unwrap());
    /// assert_eq!(div * i2 + m, i1);
    /// ```
    fn div_mod(self, rhs: Self) -> (Self, Self);

    /// Convert the integral number to an integer.
    fn to_integer(self) -> Integer;

    /// Validate whether an integral number is even.
    fn is_even(&self) -> bool { self.clone().modulus(Self::one() + Self::one()) == Self::zero() }

    /// Validate whether an integral number is odd.
    fn is_odd(&self) -> bool { !self.is_even() }
}
