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
    fn quotient(self, rhs: Self) -> Self {
        self.quot_rem(rhs).0
    }

    /// Calculating the remainder corresponding to the quotient rounded towards zero.
    fn remainder(self, rhs: Self) -> Self {
        self.quot_rem(rhs).1
    }

    /// Calculating the quotient of two integers, rounding the result towards negative infinity.
    fn division(self, rhs: Self) -> Self {
        self.div_mod(rhs).0
    }

    /// Calculating the modulus of two integers.
    fn modulus(self, rhs: Self) -> Self {
        self.div_mod(rhs).1
    }

    /// Calculating the quotient and remainder of two integers,
    /// where the quotient is rounded towards zero.
    ///
    /// ```rust,ignore
    /// let m: I, n: I;
    /// assert_eq!(m.quotient(n) * n + m.remainder(n), m);
    /// ```
    fn quot_rem(self, rhs: Self) -> (Self, Self);

    /// Calculating the division and modulus of two integers,
    /// where the division result is rounded towards negative infinity.
    ///
    /// ```rust,ignore
    /// let x: I, y: I;
    /// assert_eq!(x.division(y) * y + x.modulus(y), x);
    /// ```
    fn div_mod(self, rhs: Self) -> (Self, Self);

    /// Convert the integral number to an integer.
    fn to_integer(self) -> Integer;

    /// Validate whether an integral number is even.
    fn is_even(&self) -> bool {
        self.clone().modulus(Self::one() + Self::one()) == Self::zero()
    }

    /// Validate whether an integral number is odd.
    fn is_odd(&self) -> bool {
        !self.is_even()
    }
}
