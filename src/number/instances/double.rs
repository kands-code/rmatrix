//! # Number Type :: Double
//!
//! f64 wrapper.

use crate::number::{
    instances::{int::Int, integer::Integer, ratio::Rational},
    traits::{
        floating::Floating, fractional::Fractional, integral::Integral, number::Number, one::One,
        real::Real, realfloat::RealFloat, realfrac::RealFrac, zero::Zero,
    },
    utils::from_integeral,
};

use rand::{
    distributions::uniform::{SampleBorrow, SampleUniform, UniformFloat, UniformSampler},
    Rng,
};

#[derive(Clone, PartialOrd)]
pub struct Double {
    inner: f64,
}

impl Double {
    pub const fn of(num: f64) -> Self {
        Self { inner: num }
    }

    pub fn of_str(float_number: &str) -> Option<Self> {
        std::str::FromStr::from_str(float_number).ok()
    }
}

impl Zero for Double {
    fn zero() -> Self {
        Self { inner: 0.0f64 }
    }

    fn is_zero(&self) -> bool {
        // Use half-precision to avoid certain floating-point precision errors.
        self.inner.abs() <= core::f64::EPSILON.sqrt()
    }
}

impl One for Double {
    fn one() -> Self {
        Self { inner: 1.0f64 }
    }

    fn is_one(&self) -> bool {
        (self.clone() - Self::one()).is_zero()
    }
}

impl std::default::Default for Double {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::ops::Neg for Double {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { inner: -self.inner }
    }
}

impl std::ops::Add for Double {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

impl std::ops::Sub for Double {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner - rhs.inner,
        }
    }
}

impl std::ops::Mul for Double {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner * rhs.inner,
        }
    }
}

impl std::ops::Div for Double {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner / rhs.inner,
        }
    }
}

impl std::cmp::PartialEq for Double {
    fn eq(&self, other: &Self) -> bool {
        (self.clone() - other.clone()).is_zero()
    }
}

impl Number for Double {
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
                .parse::<f64>()
                .expect(&format!(
                    "Error[Double::from_Integer]: ({}) should be a valid f64 number.",
                    integer_number
                ));
            Self { inner }
        }
    }
}

impl RealFloat for Double {
    const FLOAT_DIGITS: Int = Int::of(53);

    const FLOAT_RANGE: (Int, Int) = (Int::of(-1021), Int::of(1024));

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

impl RealFrac for Double {
    fn proper_fraction<I: Integral>(self) -> (I, Self) {
        (
            from_integeral(
                Integer::of_str(&format!("{}", self.inner.trunc() as i64))
                    .expect("Error[Double::proper_fraction]: Should be a valid i64 number."),
            ),
            Self::of(self.inner.fract()),
        )
    }
}

impl Real for Double {
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
                    "Error[Double::to_rational]: Each digit should be within the range [1, 9] ({:?}).",
                    numerator
                )),
                Integer::of(true, &denominator).expect(&format!(
                    "Error[Double::to_rational]: Each digit should be within the range [1, 9] ({:?}).",
                    denominator
                )),
            )
            .refine()
        }
    }
}

impl Floating for Double {
    const ZERO: Self = Self { inner: 0.0f64 };

    const PI: Self = Self::of(core::f64::consts::PI);

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

impl Fractional for Double {
    fn half() -> Self {
        Self { inner: 0.5f64 }
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

impl std::fmt::Display for Double {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl std::fmt::Debug for Double {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+}", self.inner)
    }
}

impl std::str::FromStr for Double {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.trim();
        if let Ok(num) = trimmed_s.parse::<f64>() {
            Ok(Self { inner: num })
        } else {
            eprintln!(
                "Error[Double::from_str]: ({}) is not a valid Double literal.",
                trimmed_s
            );
            Err(())
        }
    }
}

/// Uniform for Double
pub struct UniformF64(UniformFloat<f64>);

impl UniformSampler for UniformF64 {
    type X = Double;

    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(UniformFloat::<f64>::new(
            low.borrow().inner,
            high.borrow().inner,
        ))
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(UniformFloat::<f64>::new_inclusive(
            low.borrow().inner,
            high.borrow().inner,
        ))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        Self::X::of(self.0.sample(rng))
    }
}

impl SampleUniform for Double {
    type Sampler = UniformF64;
}
