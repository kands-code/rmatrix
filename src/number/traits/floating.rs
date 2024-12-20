//! # traits::floating
//!
//! Types that implement this trait can be considered as floating-point numbers.

use crate::number::traits::fractional::Fractional;

/// Concepts of Floating.
pub trait Floating: Fractional {
    /// Zero of the floating-point type.
    const ZERO: Self;

    /// Floating-point constant pi, approximately `3.141592653589793`.
    const PI: Self;

    /// Exponential function with base of the Euler's number, exp(x).
    fn exponential(self) -> Self;

    /// Logarithmic function with base of the Euler's number, ln(x).
    fn logarithmic(self) -> Self;

    /// Square root function.
    fn square_root(self) -> Self {
        self.power(Self::half())
    }

    /// Sine function, sin(x).
    fn sine(self) -> Self;

    /// Cosine function, cos(x).
    fn cosine(self) -> Self;

    /// Tangent function, tan(x).
    fn tangent(self) -> Self {
        self.clone().sine() / self.cosine()
    }

    /// Arcsine function, asin(x).
    fn arc_sine(self) -> Self;

    /// Arccosine function, acos(x).
    fn arc_cosine(self) -> Self;

    /// Arctangent function, atan(x).
    fn arc_tangent(self) -> Self;

    /// Hyperbolic sine function, sinh(x).
    fn hyperbolic_sine(self) -> Self;

    /// Hyperbolic cosine function, cosh(x).
    fn hyperbolic_cosine(self) -> Self;

    /// Hyperbolic tangent function, tanh(x).
    fn hyperbolic_tangent(self) -> Self {
        self.clone().hyperbolic_sine() / self.hyperbolic_cosine()
    }

    /// Inverse hyperbolic sine function, asinh(x).
    fn arc_hyperbolic_sine(self) -> Self;

    /// Inverse hyperbolic cosine function, acosh(x).
    fn arc_hyperbolic_cosine(self) -> Self;

    /// Inverse hyperbolic tangent function, atanh(x).
    fn arc_hyperbolic_tangent(self) -> Self;

    /// Power function, pow(x, a).
    fn power(self, exponents: Self) -> Self {
        (self.logarithmic() * exponents).exponential()
    }

    /// Logarithm function with base a, log(a, x).
    fn logarithmic_base(self, base: Self) -> Self {
        self.logarithmic() / base.logarithmic()
    }

    /// Logarithm function with base Euler's number plus one, ln(1 + x).
    fn logarithmic_1p(self) -> Self {
        (Self::one() + self).logarithmic()
    }

    /// Exponential function with base Euler's number minus one, exp(x) - 1.
    fn exponential_1m(self) -> Self {
        self.exponential() - Self::one()
    }

    /// Logarithm function plus one of the exponential function
    /// with base Euler's number, ln(1 + exp(x)).
    fn logarithmic_1p_exponential(self) -> Self {
        self.exponential().logarithmic_1p()
    }

    /// Logarithm function minus one of the exponential function
    /// with base Euler's number, ln(1 - exp(x)).
    fn logarithmic_1m_exponential(self) -> Self {
        (-self.exponential()).logarithmic_1p()
    }
}
