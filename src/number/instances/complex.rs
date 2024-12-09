//! # Number Type :: Complex
//!
//! Complex number for RealFloat number.

use crate::number::{
    instances::{integer::Integer, ratio::Rational},
    traits::{
        floating::Floating, fractional::Fractional, number::Number, one::One, realfloat::RealFloat,
        zero::Zero,
    },
};

#[derive(Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_mat", derive(serde::Deserialize, serde::Serialize))]
pub struct Complex<F: RealFloat> {
    pub real: F,
    pub imaginary: F,
}

impl<F: RealFloat> Complex<F> {
    pub const fn of(real: F, imaginary: F) -> Self {
        Self { real, imaginary }
    }

    pub fn of_str(float_number: &str) -> Result<Self, String> {
        std::str::FromStr::from_str(float_number)
    }

    pub fn conjugate(self) -> Self {
        Self {
            real: self.real,
            imaginary: -self.imaginary,
        }
    }

    pub fn norm(self) -> F {
        (self.real.clone() * self.real + self.imaginary.clone() * self.imaginary).square_root()
    }
}

impl<F: RealFloat> Zero for Complex<F> {
    fn zero() -> Self {
        Self {
            real: F::zero(),
            imaginary: F::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.imaginary.is_zero()
    }
}

impl<F: RealFloat> One for Complex<F> {
    fn one() -> Self {
        Self {
            real: F::one(),
            imaginary: F::zero(),
        }
    }

    fn is_one(&self) -> bool {
        self.real.is_one() && self.imaginary.is_zero()
    }
}

impl<F: RealFloat> std::ops::Neg for Complex<F> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            real: -self.real,
            imaginary: -self.imaginary,
        }
    }
}

impl<F: RealFloat> std::ops::Add for Complex<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real + rhs.real,
            imaginary: self.imaginary + rhs.imaginary,
        }
    }
}

impl<F: RealFloat> std::ops::Sub for Complex<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real - rhs.real,
            imaginary: self.imaginary - rhs.imaginary,
        }
    }
}

impl<F: RealFloat> std::ops::Mul for Complex<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real.clone() * rhs.real.clone()
                - self.imaginary.clone() * rhs.imaginary.clone(),
            imaginary: self.real * rhs.imaginary + self.imaginary * rhs.real,
        }
    }
}

impl<F: RealFloat> std::ops::Div for Complex<F> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let denominator =
            rhs.real.clone() * rhs.real.clone() + rhs.imaginary.clone() * rhs.imaginary.clone();
        let numerator = self * rhs.conjugate();
        Self {
            real: numerator.real / denominator.clone(),
            imaginary: numerator.imaginary / denominator,
        }
    }
}

impl<F: RealFloat> Number for Complex<F> {
    fn absolute_value(&self) -> Self {
        Self {
            real: self.real.absolute_value(),
            imaginary: self.imaginary.absolute_value(),
        }
    }

    fn sign_number(&self) -> Self {
        if self.is_zero() {
            Self::zero()
        } else {
            let n = self.clone().norm();
            Self {
                real: self.real.sign_number() / n.clone(),
                imaginary: self.imaginary.sign_number() / n,
            }
        }
    }

    fn from_integer(integer_number: Integer) -> Self {
        if integer_number.is_zero() {
            Self::zero()
        } else {
            Self {
                real: F::from_integer(integer_number),
                imaginary: F::zero(),
            }
        }
    }
}

impl<F: RealFloat> Floating for Complex<F> {
    const ZERO: Self = Self {
        real: F::ZERO,
        imaginary: F::ZERO,
    };

    const PI: Self = Self {
        real: F::PI,
        imaginary: F::ZERO,
    };

    fn exponential(self) -> Self {
        Self {
            real: self.real.clone().exponential() * self.imaginary.clone().cosine(),
            imaginary: self.real.exponential() * self.imaginary.sine(),
        }
    }

    fn logarithmic(self) -> Self {
        let n = self.clone().norm();
        Self {
            real: n.logarithmic(),
            imaginary: (self.imaginary / self.real).arc_tangent(),
        }
    }

    fn sine(self) -> Self {
        Self {
            real: self.real.clone().sine() * self.imaginary.clone().hyperbolic_cosine(),
            imaginary: self.real.cosine() * self.imaginary.hyperbolic_sine(),
        }
    }

    fn cosine(self) -> Self {
        Self {
            real: self.real.clone().cosine() * self.imaginary.clone().hyperbolic_cosine(),
            imaginary: -self.real.sine() * self.imaginary.hyperbolic_sine(),
        }
    }

    fn arc_sine(self) -> Self {
        let i = Self {
            real: F::zero(),
            imaginary: F::one(),
        };
        -i.clone()
            * (i * self.clone() + (Self::one() - self.clone() * self).square_root()).logarithmic()
    }

    fn arc_cosine(self) -> Self {
        let i = Self {
            real: F::zero(),
            imaginary: F::one(),
        };
        -i.clone()
            * (self.clone() + i * (Self::one() - self.clone() * self).square_root()).logarithmic()
    }

    fn arc_tangent(self) -> Self {
        todo!()
    }

    fn hyperbolic_sine(self) -> Self {
        (self.clone().exponential() - (-self).exponential()) / (Self::one() + Self::one())
    }

    fn hyperbolic_cosine(self) -> Self {
        (self.clone().exponential() + (-self).exponential()) / (Self::one() + Self::one())
    }

    fn arc_hyperbolic_sine(self) -> Self {
        todo!()
    }

    fn arc_hyperbolic_cosine(self) -> Self {
        todo!()
    }

    fn arc_hyperbolic_tangent(self) -> Self {
        todo!()
    }
}

impl<F: RealFloat> Fractional for Complex<F> {
    fn half() -> Self {
        Self {
            real: F::half(),
            imaginary: F::zero(),
        }
    }

    fn reciprocal(self) -> Self {
        let n = self.clone().norm();
        let conj = self.conjugate();
        Self {
            real: conj.real / n.clone(),
            imaginary: conj.imaginary / n,
        }
    }

    fn from_rational(rational_number: Rational) -> Self {
        Self::from_integer(rational_number.numerator)
            / Self::from_integer(rational_number.denominator)
    }
}

impl<F: RealFloat> std::fmt::Display for Complex<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} :+ {}", self.real, self.imaginary)
    }
}

impl<F: RealFloat> std::fmt::Debug for Complex<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+} :+ {:+}", self.real, self.imaginary)
    }
}

impl<F: RealFloat> std::str::FromStr for Complex<F> {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.split(":+").map(|p| p.trim()).collect::<Vec<_>>();
        if trimmed_s.len() != 2 {
            Err(format!(
                "Error[Complex::from_str]: {} is not a valid complex number",
                s
            ))
        } else {
            let real = F::from_str(trimmed_s[0])?;
            let imaginary = F::from_str(trimmed_s[1])?;
            Ok(Self { real, imaginary })
        }
    }
}
