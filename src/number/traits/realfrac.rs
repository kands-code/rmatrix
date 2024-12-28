//! # traits::realfrac
//!
//! Types that implement this trait can be considered as real fractional numbers.

use crate::number::traits::{fractional::Fractional, integral::Integral, real::Real};

/// Concepts of RealFrac.
pub trait RealFrac: Real + Fractional {
    /// Extract the integral part and the fractional part from a real fractional number.
    fn proper_fraction<I: Integral>(self) -> (I, Self);

    /// Return the nearest integral number between itself and zero.
    fn truncate<I: Integral>(self) -> I { self.proper_fraction().0 }

    /// Return the nearest integral number to itself.
    fn round<I: Integral>(self) -> I {
        let (integral_part, fractional_part): (I, Self) = self.proper_fraction();
        let integral_part_acc = if fractional_part < Self::zero() {
            integral_part.clone() - I::one()
        } else {
            integral_part.clone() + I::one()
        };
        match (fractional_part.absolute_value() - Self::half()).sign_number() {
            s if s == Self::one() => integral_part_acc,
            s if s == Self::zero() => {
                if integral_part.is_even() {
                    integral_part
                } else {
                    integral_part_acc
                }
            }
            _ => integral_part,
        }
    }

    /// Return the smallest integer greater than itself.
    fn ceiling<I: Integral>(self) -> I {
        let (integral_part, fractional_part) = self.proper_fraction();
        if fractional_part > Self::zero() {
            integral_part + I::one()
        } else {
            integral_part
        }
    }

    /// Return the largest integer less than itself.
    fn floor<I: Integral>(self) -> I {
        let (integral_part, fractional_part) = self.proper_fraction();
        if fractional_part < Self::zero() {
            integral_part - I::zero()
        } else {
            integral_part
        }
    }
}
