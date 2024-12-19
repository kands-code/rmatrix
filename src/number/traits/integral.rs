//! # Number Trait :: Integral
//!
//! A type that implements this trait is a integral number.

use crate::number::{instances::integer::Integer, traits::real::Real};

/// Integral
pub trait Integral: Real
where
    Self: std::cmp::Ord,
{
    /// integer division truncated toward zero
    fn quotient(self, rhs: Self) -> Self {
        self.quot_rem(rhs).0
    }

    /// integer remainder
    fn remainder(self, rhs: Self) -> Self {
        self.quot_rem(rhs).1
    }

    /// integer division truncated toward negative infinity
    fn division(self, rhs: Self) -> Self {
        self.div_mod(rhs).0
    }

    /// integer modulus
    fn modulus(self, rhs: Self) -> Self {
        self.div_mod(rhs).1
    }

    /// simultaneous 'quotient' and 'remainder'
    ///
    /// ```rust,ignore
    /// assert_eq!(
    ///     integral_number.clone()
    ///         .quotient(m.clone())
    ///         .mul(m.clone()) + n.remainder(m),
    ///     integral_number
    /// )
    /// ```
    fn quot_rem(self, rhs: Self) -> (Self, Self);

    /// simultaneous 'division' and 'modulus'
    ///
    /// ```rust,ignore
    /// assert_eq!(
    ///     integral_number_x.clone()
    ///         .division(integral_number_y.clone())
    ///         .mul(integral_number_y.clone())
    ///         + integral_number_x.modulus(integral_number_y),
    ///     integral_number_x
    /// )
    /// ```
    fn div_mod(self, rhs: Self) -> (Self, Self);

    /// convert an intgeral number to an integer
    fn to_integer(self) -> Integer;

    /// determine whether an integral number is even
    fn is_even(&self) -> bool {
        self.clone().modulus(Self::one() + Self::one()) == Self::zero()
    }

    /// determine whether an integral number is odd
    fn is_odd(&self) -> bool {
        !self.is_even()
    }
}
