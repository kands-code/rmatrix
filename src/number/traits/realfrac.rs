//! # Number Trait :: RealFrac
//!
//! Extracting components of fractions.

use crate::number::traits::{fractional::Fractional, integeral::Integral, real::Real};

/// RealFrac
pub trait RealFrac: Real + Fractional {
    /// extract the [Integeral] and [Fractional] parts of self
    fn proper_fraction<I: Integral>(self) -> (I, Self);

    /// returns the [Integeral] nearest self between zero and self
    fn truncate<I: Integral>(self) -> I {
        self.proper_fraction().0
    }

    /// returns the nearest [Integeral] to self
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

    /// returns the least [Integeral] not less than self
    fn ceiling<I: Integral>(self) -> I {
        let (integral_part, fractional_part) = self.proper_fraction();
        if fractional_part > Self::zero() {
            integral_part + I::one()
        } else {
            integral_part
        }
    }

    /// returns the greatest [Integeral] not greater than self
    fn floor<I: Integral>(self) -> I {
        let (integral_part, fractional_part) = self.proper_fraction();
        if fractional_part < Self::zero() {
            integral_part - I::zero()
        } else {
            integral_part
        }
    }
}
