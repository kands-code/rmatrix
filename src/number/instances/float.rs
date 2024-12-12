//! # Number Type :: Float
//!
//! f32 wrapper.

use crate::number::{
    instances::{int::Int, integer::Integer, ratio::Rational},
    traits::{
        floating::Floating, fractional::Fractional, integral::Integral, number::Number, one::One,
        real::Real, realfloat::RealFloat, realfrac::RealFrac, zero::Zero,
    },
    utils::from_integeral,
};

#[cfg(feature = "rand_mat")]
use rand::{
    distributions::uniform::{SampleBorrow, SampleUniform, UniformFloat, UniformSampler},
    Rng,
};

#[derive(Clone, PartialEq, PartialOrd)]
pub struct Float {
    inner: f32,
}

impl Float {
    pub const fn of(num: f32) -> Self {
        Self { inner: num }
    }

    pub fn of_str(float_number: &str) -> Result<Self, String> {
        std::str::FromStr::from_str(float_number)
    }
}

impl Zero for Float {
    fn zero() -> Self {
        Self { inner: 0.0f32 }
    }

    fn is_zero(&self) -> bool {
        self.inner.abs() <= f32::EPSILON
    }
}

impl One for Float {
    fn one() -> Self {
        Self { inner: 1.0f32 }
    }

    fn is_one(&self) -> bool {
        (self.clone() - Self::one()).is_zero()
    }
}

impl std::default::Default for Float {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::ops::Neg for Float {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { inner: -self.inner }
    }
}

impl std::ops::Add for Float {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

impl std::ops::Sub for Float {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner - rhs.inner,
        }
    }
}

impl std::ops::Mul for Float {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner * rhs.inner,
        }
    }
}

impl std::ops::Div for Float {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner / rhs.inner,
        }
    }
}

impl Number for Float {
    fn absolute_value(&self) -> Self {
        Self {
            inner: self.inner.abs(),
        }
    }

    fn sign_number(&self) -> Self {
        if self.is_zero() {
            Self::zero()
        } else if self.inner.is_sign_positive() {
            Self::one()
        } else {
            -Self::one()
        }
    }

    fn from_integer(integer_number: Integer) -> Self {
        if integer_number.is_zero() {
            Self::zero()
        } else {
            let inner = format!("{}", integer_number)
                .parse::<f32>()
                .expect("Error[Float::from<Integer>]: integer_number should be a valid f32 number");
            Self { inner }
        }
    }
}

impl RealFloat for Float {
    const FLOAT_DIGITS: Int = Int::of(24);

    const FLOAT_RANGE: (Int, Int) = (Int::of(-125), Int::of(128));

    fn is_not_a_number(&self) -> bool {
        self.inner.is_nan()
    }

    fn is_infinite_number(&self) -> bool {
        self.inner.is_infinite()
    }

    fn is_denormalized(&self) -> bool {
        self.inner.is_subnormal()
    }

    fn is_negative_zero(&self) -> bool {
        self.is_zero() && self.inner.is_sign_negative()
    }
}

impl RealFrac for Float {
    fn proper_fraction<I: Integral>(self) -> (I, Self) {
        (
            from_integeral(Int::of(self.inner.trunc() as i32)),
            Self::of(self.inner.fract()),
        )
    }
}

impl Real for Float {
    fn to_rational(self) -> Rational {
        if self.is_not_a_number() || self.is_infinite_number() {
            panic!(
                "Error[Float::to_rational]: {} is not a valid floating number",
                self
            );
        } else if self.is_zero() {
            Rational {
                numerator: Integer::zero(),
                denominator: Integer::one(),
            }
        } else {
            let sign = !self.inner.is_sign_negative();
            let str_inner = format!("{}", self.inner);
            let p = str_inner.split(".").collect::<Vec<&str>>()[1].len();
            let mut denominator = vec![0u8; p + 1];
            denominator[0] = 1u8;
            let numerator = str_inner
                .chars()
                .filter(|&c| c.is_ascii_digit())
                .map(|c| c as u8 - '0' as u8)
                .collect::<Vec<u8>>();
            Rational::of(
                Integer::of(sign, &numerator).expect(&format!(
                    "Error[Float::to_rational]: every digits should be in [0, 10) {:?}",
                    numerator
                )),
                Integer::of(true, &denominator).expect(&format!(
                    "Error[Float::to_rational]: every digits should be in [0, 10) {:?}",
                    denominator
                )),
            )
            .refine()
        }
    }
}

impl Floating for Float {
    const ZERO: Self = Self { inner: 0.0f32 };

    const PI: Self = Self::of(core::f32::consts::PI);

    fn exponential(self) -> Self {
        Self {
            inner: self.inner.exp(),
        }
    }

    fn logarithmic(self) -> Self {
        Self {
            inner: self.inner.ln(),
        }
    }

    fn sine(self) -> Self {
        Self {
            inner: self.inner.sin(),
        }
    }

    fn cosine(self) -> Self {
        Self {
            inner: self.inner.cos(),
        }
    }

    fn arc_sine(self) -> Self {
        Self {
            inner: self.inner.asin(),
        }
    }

    fn arc_cosine(self) -> Self {
        Self {
            inner: self.inner.acos(),
        }
    }

    fn arc_tangent(self) -> Self {
        Self {
            inner: self.inner.atan(),
        }
    }

    fn hyperbolic_sine(self) -> Self {
        Self {
            inner: self.inner.sinh(),
        }
    }

    fn hyperbolic_cosine(self) -> Self {
        Self {
            inner: self.inner.cosh(),
        }
    }

    fn arc_hyperbolic_sine(self) -> Self {
        Self {
            inner: self.inner.asinh(),
        }
    }

    fn arc_hyperbolic_cosine(self) -> Self {
        Self {
            inner: self.inner.acosh(),
        }
    }

    fn arc_hyperbolic_tangent(self) -> Self {
        Self {
            inner: self.inner.atanh(),
        }
    }
}

impl Fractional for Float {
    fn half() -> Self {
        Self { inner: 0.5f32 }
    }

    fn reciprocal(self) -> Self {
        let rational = self.to_rational();
        Self::from_integer(rational.denominator) / Self::from_integer(rational.numerator)
    }

    fn from_rational(rational_number: Rational) -> Self {
        Self::from_integer(rational_number.numerator)
            / Self::from_integer(rational_number.denominator)
    }
}

impl std::fmt::Display for Float {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl std::fmt::Debug for Float {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+}", self.inner)
    }
}

impl std::str::FromStr for Float {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.trim();
        if let Ok(num) = trimmed_s.parse::<f32>() {
            Ok(Self { inner: num })
        } else {
            Err(format!(
                "Error[Float::from_str]: {} is not a valid float number",
                trimmed_s
            ))
        }
    }
}

#[cfg(feature = "rand_mat")]
#[doc(cfg(feature = "rand_mat"))]
/// Uniform for Float
pub struct UniformF32(UniformFloat<f32>);

#[cfg(feature = "rand_mat")]
impl UniformSampler for UniformF32 {
    type X = Float;

    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(UniformFloat::<f32>::new(
            low.borrow().inner,
            high.borrow().inner,
        ))
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(UniformFloat::<f32>::new_inclusive(
            low.borrow().inner,
            high.borrow().inner,
        ))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        Self::X::of(self.0.sample(rng))
    }
}

#[cfg(feature = "rand_mat")]
impl SampleUniform for Float {
    type Sampler = UniformF32;
}
