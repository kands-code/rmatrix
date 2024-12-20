//! # Number Trait :: Floating
//!
//! A type that implements this trait is a floating number.

use crate::number::traits::fractional::Fractional;

/// Floating
pub trait Floating: Fractional {
    /// zero
    const ZERO: Self;

    /// pi \approx 3.141593
    const PI: Self;

    /// exp(x)
    fn exponential(self) -> Self;

    /// ln(x)
    fn logarithmic(self) -> Self;

    /// sqrt(x)
    fn square_root(self) -> Self {
        self.power(Self::half())
    }

    /// sin(x)
    fn sine(self) -> Self;

    /// cos(x)
    fn cosine(self) -> Self;

    /// tan(x)
    fn tangent(self) -> Self {
        self.clone().sine() / self.cosine()
    }

    /// asin(x)
    fn arc_sine(self) -> Self;

    /// acos(x)
    fn arc_cosine(self) -> Self;

    /// atan(x)
    fn arc_tangent(self) -> Self;

    /// sinh(x)
    fn hyperbolic_sine(self) -> Self;

    /// cosh(x)
    fn hyperbolic_cosine(self) -> Self;

    /// tanh(x)
    fn hyperbolic_tangent(self) -> Self {
        self.clone().hyperbolic_sine() / self.hyperbolic_cosine()
    }

    /// asinh(x)
    fn arc_hyperbolic_sine(self) -> Self;

    /// acosh(x)
    fn arc_hyperbolic_cosine(self) -> Self;

    /// atanh(x)
    fn arc_hyperbolic_tangent(self) -> Self;

    /// x^y
    fn power(self, exponents: Self) -> Self {
        (self.logarithmic() * exponents).exponential()
    }

    /// log(a, x)
    fn logarithmic_base(self, base: Self) -> Self {
        self.logarithmic() / base.logarithmic()
    }

    /// log(1 + x)
    fn logarithmic_1p(self) -> Self {
        (Self::one() + self).logarithmic()
    }

    /// exp(x) - 1
    fn exponential_1m(self) -> Self {
        self.exponential() - Self::one()
    }

    /// log(1 + exp(x))
    fn logarithmic_1p_exponential(self) -> Self {
        self.exponential().logarithmic_1p()
    }

    /// log(1 - exp(x))
    fn logarithmic_1m_exponential(self) -> Self {
        (-self.exponential()).logarithmic_1p()
    }
}
