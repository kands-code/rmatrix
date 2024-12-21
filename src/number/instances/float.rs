//! # instances::float
//!
//! Functions and related implementations for single precision floating-point numbers.

use crate::number::{
    instances::{int::Int, integer::Integer, ratio::Rational},
    traits::{
        floating::Floating, fractional::Fractional, integral::Integral, number::Number, one::One,
        real::Real, realfloat::RealFloat, realfrac::RealFrac, zero::Zero,
    },
    utils::from_integral,
};

use rand::{
    distributions::uniform::{SampleBorrow, SampleUniform, UniformFloat, UniformSampler},
    Rng,
};

/// Float numbers are the wrapper type for f32.
#[derive(Clone, PartialOrd)]
pub struct Float {
    inner: f32,
}

impl Float {
    /// Construct float numbers from f32.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::float::Float;
    ///
    /// fn main() {
    ///     let _f = Float::of(12.0);
    /// }
    /// ```
    pub const fn of(num: f32) -> Self {
        Self { inner: num }
    }

    /// Construct float numbers from string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::float::Float;
    ///
    /// fn main() {
    ///     let sf = Float::of_str("-12.0").unwrap();
    ///     let f = Float::of(-12.0);
    ///     assert_eq!(sf, f);
    /// }
    /// ```
    pub fn of_str(float_number: &str) -> Option<Self> {
        std::str::FromStr::from_str(float_number).ok()
    }
}

/// Implement the concept of ZERO for the float number.
impl Zero for Float {
    fn zero() -> Self {
        Self { inner: 0.0f32 }
    }

    /// Validate whether a float number is ZERO.
    ///
    /// Use half-precision to avoid certain floating-point precision errors.
    ///
    /// ```rust
    /// use rmatrix_ks::number::{instances::float::Float, traits::zero::Zero};
    ///
    /// fn main() {
    ///     let f = Float::of(core::f32::EPSILON);
    ///     assert!(f.is_zero());
    /// }
    /// ```
    fn is_zero(&self) -> bool {
        self.inner.abs() <= core::f32::EPSILON.sqrt()
    }
}

/// Implement the concept of ONE for the float number.
impl One for Float {
    fn one() -> Self {
        Self { inner: 1.0f32 }
    }

    fn is_one(&self) -> bool {
        (self.clone() - Self::one()).is_zero()
    }
}

/// Implement Default for the float number.
impl std::default::Default for Float {
    fn default() -> Self {
        Self::zero()
    }
}

/// Implement the negation operation for the float number.
impl std::ops::Neg for Float {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { inner: -self.inner }
    }
}

/// Implement the addition operation for the float number.
impl std::ops::Add for Float {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

/// Implement the subtraction operation for the float number.
impl std::ops::Sub for Float {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner - rhs.inner,
        }
    }
}

/// Implement the multiplication operation for the float number.
impl std::ops::Mul for Float {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner * rhs.inner,
        }
    }
}

/// Implement the division operation for the float number.
impl std::ops::Div for Float {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner / rhs.inner,
        }
    }
}

/// Implement equality for float numbers.
impl std::cmp::PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        (self.clone() - other.clone()).is_zero()
    }
}

/// Implement the concept of NUMBER for the float number.
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

    /// Construct a float number from an integer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{float::Float, integer::Integer},
    ///     traits::number::Number,
    /// };
    ///
    /// fn main() {
    ///     let i1 = Integer::of_str("123456789").unwrap();
    ///     let f1 = Float::from_integer(i1);
    ///     assert_eq!(f1, Float::of(123456789.0));
    /// }
    /// ```
    ///
    /// ## Warnings
    ///
    /// <div class="warning">
    ///
    /// When the size of an integer exceeds the maximum integer
    /// representable by a single-precision floating-point number,
    /// significant errors may occur.
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{float::Float, integer::Integer},
    ///     traits::number::Number,
    /// };
    ///
    /// fn main() {
    ///     let i2 = Integer::of_str("1234567891011121314151617181920").unwrap();
    ///     let f2 = Float::from_integer(i2);
    ///     assert_eq!(
    ///         f2,
    ///         Float::of(1234567900000000000000000000000.0)
    ///     );
    /// }
    /// ```
    ///
    /// </div>
    fn from_integer(integer_number: Integer) -> Self {
        if integer_number.is_zero() {
            Self::zero()
        } else {
            let inner = format!("{:?}", integer_number)
                .parse::<f32>()
                .expect(&format!(
                    "Error[Float::from_Integer]: ({}) should be a valid f32 number.",
                    integer_number
                ));
            Self { inner }
        }
    }
}

/// Implement the concept of RealFloat for Float.
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

/// Implement the concept of RealFrac for Float.
impl RealFrac for Float {
    fn proper_fraction<I: Integral>(self) -> (I, Self) {
        (
            from_integral(Int::of(self.inner.trunc() as i32)),
            Self::of(self.inner.fract()),
        )
    }
}

/// Implement the concept of Real for Float.
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
            let str_inner = format!("{:?}", self.inner);
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
                    concat!(
                        "Error[Float::to_rational]: ",
                        "Each digit should be within the range [1, 9] ({:?})."
                    ),
                    numerator
                )),
                Integer::of(true, &denominator).expect(&format!(
                    concat!(
                        "Error[Float::to_rational]: ",
                        "Each digit should be within the range [1, 9] ({:?})."
                    ),
                    denominator
                )),
            )
            .refine()
        }
    }
}

/// Implement the concept of Floating for Float.
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

/// Implement the concept of Fractional for Float.
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

/// Implement Display for Float.
impl std::fmt::Display for Float {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

/// Implement Debug for Float.
impl std::fmt::Debug for Float {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+}", self.inner)
    }
}

/// Implement FromStr for Float.
impl std::str::FromStr for Float {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.trim();
        if let Ok(num) = trimmed_s.parse::<f32>() {
            Ok(Self { inner: num })
        } else {
            eprintln!(
                "Error[Float::from_str]: ({}) is not a valid Float literal.",
                trimmed_s
            );
            Err(())
        }
    }
}

/// Uniform distribution of float numbers.
pub struct UniformF32(UniformFloat<f32>);

/// Implement uniform sampling for the uniform distribution of float numbers.
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

/// Implement uniform sampling for float numbers.
impl SampleUniform for Float {
    type Sampler = UniformF32;
}
