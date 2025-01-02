//! # instances::double
//!
//! Functions and related implementations for double precision floating-point numbers.

use rand::{
    Rng,
    distributions::uniform::{SampleBorrow, SampleUniform, UniformFloat, UniformSampler},
};

use crate::number::{
    instances::{int::Int, integer::Integer, ratio::Rational},
    traits::{
        floating::Floating,
        fractional::Fractional,
        integral::Integral,
        number::Number,
        one::One,
        real::Real,
        realfloat::RealFloat,
        realfrac::RealFrac,
        zero::Zero,
    },
    utils::{from_integral, non_negative_integral_power},
};

/// Double numbers are the wrapper type for f64.
#[derive(Clone, PartialOrd)]
pub struct Double {
    /// Internal data.
    inner: f64,
}

impl Double {
    /// Construct double numbers from f64.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::double::Double;
    ///
    /// fn main() { let _d = Double::of(12.0); }
    /// ```
    pub const fn of(num: f64) -> Self { Self { inner: num } }

    /// Construct double numbers from string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{instances::double::Double, traits::zero::Zero};
    ///
    /// fn main() {
    ///     let sd = Double::of_str("-12.0").unwrap();
    ///     let d = Double::of(-12.0);
    ///     assert!((sd - d).is_zero());
    /// }
    /// ```
    pub fn of_str(float_number: &str) -> Option<Self> {
        std::str::FromStr::from_str(float_number).ok()
    }
}

/// Implement the concept of ZERO for the double number.
impl Zero for Double {
    fn zero() -> Self { Self { inner: 0.0f64 } }

    /// Validate whether a double number is ZERO.
    ///
    /// Use `eps = 1.52587890625e-5` to avoid certain floating-point precision errors.
    ///
    /// ```rust
    /// use rmatrix_ks::number::{instances::double::Double, traits::zero::Zero};
    ///
    /// fn main() {
    ///     let d = Double::of(core::f64::EPSILON);
    ///     assert!(d.is_zero());
    /// }
    /// ```
    fn is_zero(&self) -> bool { self.inner.abs() <= 1.52587890625e-5 }
}

/// Implement the concept of ONE for the double number.
impl One for Double {
    fn one() -> Self { Self { inner: 1.0f64 } }

    fn is_one(&self) -> bool { (self.clone() - Self::one()).is_zero() }
}

/// Implement Default for the double number.
impl std::default::Default for Double {
    fn default() -> Self { Self::zero() }
}

/// Implement the negation operation for the double number.
impl std::ops::Neg for Double {
    type Output = Self;

    fn neg(self) -> Self::Output { Self { inner: -self.inner } }
}

/// Implement the addition operation for the double number.
impl std::ops::Add for Double {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

/// Implement the subtraction operation for the double number.
impl std::ops::Sub for Double {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner - rhs.inner,
        }
    }
}

/// Implement the multiplication operation for the double number.
impl std::ops::Mul for Double {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner * rhs.inner,
        }
    }
}

/// Implement the division operation for the double number.
impl std::ops::Div for Double {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner / rhs.inner,
        }
    }
}

/// Implement equality for double numbers.
impl std::cmp::PartialEq for Double {
    fn eq(&self, other: &Self) -> bool { (self.clone() - other.clone()).is_zero() }
}

/// Implement the concept of NUMBER for the double number.
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

    /// Construct a double number from an integer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{double::Double, integer::Integer},
    ///     traits::number::Number,
    /// };
    ///
    /// fn main() {
    ///     let i1 = Integer::of_str("12345678910").unwrap();
    ///     let d1 = Double::from_integer(i1);
    ///     assert_eq!(d1, Double::of(12345678910.0));
    /// }
    /// ```
    ///
    /// ## Warnings
    ///
    /// <div class="warning">
    ///
    /// When the size of an integer exceeds the maximum integer
    /// representable by a double-precision floating-point number,
    /// significant errors may occur.
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{double::Double, integer::Integer},
    ///     traits::number::Number,
    /// };
    ///
    /// fn main() {
    ///     let i2 = Integer::of_str("123456789101112131415161718192021222324252627").unwrap();
    ///     let d2 = Double::from_integer(i2);
    ///     assert_eq!(
    ///         d2,
    ///         Double::of(123456789101112130000000000000000000000000000.0)
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
                .parse::<f64>()
                .expect(&format!(
                    "Error[Double::from_Integer]: ({}) should be a valid f64 number.",
                    integer_number
                ));
            Self { inner }
        }
    }
}

/// Implement the concept of RealFloat for Double.
impl RealFloat for Double {
    const FLOAT_DIGITS: Int = Int::of(53);
    const FLOAT_RANGE: (Int, Int) = (Int::of(-1021), Int::of(1024));

    fn is_not_a_number(&self) -> bool { self.inner.is_nan() }

    fn is_infinite_number(&self) -> bool { self.inner.is_infinite() }

    fn is_denormalized(&self) -> bool { self.inner.is_subnormal() }

    fn is_negative_zero(&self) -> bool { self.is_zero() && self.inner.is_sign_negative() }

    fn hypot(self, rhs: Self) -> Self {
        Self {
            inner: self.inner.hypot(rhs.inner),
        }
    }
}

/// Implement the concept of RealFrac for Double.
impl RealFrac for Double {
    fn proper_fraction<I: Integral>(self) -> (I, Self) {
        (
            from_integral(
                Integer::of_str(&format!("{:?}", self.inner.trunc() as i64))
                    .expect("Error[Double::proper_fraction]: Should be a valid i64 number."),
            ),
            Self::of(self.inner.fract()),
        )
    }
}

/// Implement the concept of Real for Double.
impl Real for Double {
    /// Convert double-precision floating-point number to rational number.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{double::Double, ratio::Rational},
    ///     traits::real::Real,
    /// };
    ///
    /// fn main() {
    ///     let m = Double::of(3.14);
    ///     let m_rat = m.to_rational();
    ///     let rat_expect = Rational::of_str("7070651414971679 % 2251799813685248").unwrap();
    ///     assert_eq!(m_rat, rat_expect);
    /// }
    /// ```
    fn to_rational(self) -> Rational {
        assert!(
            !(self.is_not_a_number() || self.is_infinite_number()),
            "Error[Double::to_rational]: {} is not a valid floating number",
            self
        );

        if self.is_zero() {
            Rational {
                numerator: Integer::zero(),
                denominator: Integer::one(),
            }
        } else {
            let (sig, exp) = self.decode_float();
            let denominator =
                non_negative_integral_power(Int::of(2).to_integer(), exp.absolute_value())
                    .expect(concat!(
                        "Error[Double::to_rational]: ",
                        "Failed to compute the denominator via exponentiation."
                    ));
            Rational::of(sig, denominator).refine()
        }
    }
}

/// Implement the concept of Floating for Double.
impl Floating for Double {
    const PI: Self = Self::of(core::f64::consts::PI);
    const ZERO: Self = Self { inner: 0.0f64 };

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

    fn hyperbolic_tangent(self) -> Self {
        Self {
            inner: self.inner.tanh(),
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

/// Implement the concept of Fractional for Double.
impl Fractional for Double {
    fn half() -> Self { Self { inner: 0.5f64 } }

    fn reciprocal(self) -> Self {
        let rational = self.to_rational();
        Self::from_integer(rational.denominator) / Self::from_integer(rational.numerator)
    }

    fn from_rational(rational_number: Rational) -> Self {
        Self::from_integer(rational_number.numerator)
            / Self::from_integer(rational_number.denominator)
    }
}

/// Implement Display for Double.
impl std::fmt::Display for Double {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

/// Implement Debug for Double.
impl std::fmt::Debug for Double {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+}", self.inner)
    }
}

/// Implement FromStr for Double.
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

/// Uniform distribution of double numbers.
pub struct UniformF64(UniformFloat<f64>);

/// Implement uniform sampling for the uniform distribution of double numbers.
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

/// Implement uniform sampling for double numbers.
impl SampleUniform for Double {
    type Sampler = UniformF64;
}
