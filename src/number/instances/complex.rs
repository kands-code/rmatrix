//! # instances::complex
//!
//! Functions and related implementations for complex numbers,
//! where both the real and imaginary parts of the complex numbers are RealFloat numbers.

use rand::{
    Rng,
    distributions::{
        Distribution,
        uniform::{SampleBorrow, SampleUniform, Uniform, UniformSampler},
    },
};

use crate::number::{
    instances::{integer::Integer, ratio::Rational},
    traits::{
        floating::Floating,
        fractional::Fractional,
        number::Number,
        one::One,
        realfloat::RealFloat,
        zero::Zero,
    },
};

/// A Complex number is a Number composed of two RealFloat numbers.
#[derive(Clone, PartialEq, Eq)]
pub struct Complex<F: RealFloat> {
    /// Real part.
    pub real: F,
    /// Imaginary part.
    pub imaginary: F,
}

impl<F: RealFloat> Complex<F> {
    /// Construct a complex number by specifying the real part and the imaginary part.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use rmatrix_ks::number::instances::{complex::Complex, double::Double};
    ///
    /// fn main() { let _c = Complex::of(Double::of(2.0), Double::of(1.0)); }
    /// ```
    pub const fn of(real: F, imaginary: F) -> Self { Self { real, imaginary } }

    /// Construct a complex number from a string.
    ///
    /// The string format is `A :+ B`,
    /// where `A` is the real part and `B` is the imaginary part.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::{complex::Complex, double::Double};
    ///
    /// fn main() {
    ///     let s = "2 :+ -1";
    ///     let c = Complex::of(Double::of(2.0), Double::of(-1.0));
    ///     assert_eq!(Complex::of_str(s), Some(c));
    /// }
    /// ```
    pub fn of_str(complex_number: &str) -> Option<Self> {
        std::str::FromStr::from_str(complex_number).ok()
    }

    /// Get its conjugate complex number.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::{complex::Complex, double::Double};
    ///
    /// fn main() {
    ///     let c = Complex::of(Double::of(2.0), Double::of(-1.0));
    ///     let cc = Complex::of(Double::of(2.0), Double::of(1.0));
    ///     assert_eq!(c.conjugate(), cc);
    /// }
    /// ```
    pub fn conjugate(self) -> Self {
        Self {
            real: self.real,
            imaginary: -self.imaginary,
        }
    }

    /// Calculate the norm of the complex number.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double},
    ///     traits::{floating::Floating, zero::Zero},
    /// };
    ///
    /// fn main() {
    ///     let c = Complex::of(Double::of(3.0), Double::of(-4.0));
    ///     assert!((c.norm() - Double::of(5.0)).is_zero());
    /// }
    /// ```
    pub fn norm(self) -> F {
        (self.real.clone() * self.real + self.imaginary.clone() * self.imaginary).square_root()
    }
}

/// Implement the concept of ZERO for the complex number.
impl<F: RealFloat> Zero for Complex<F> {
    /// Return the ZERO complex number,
    /// where both the real and imaginary parts are zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double},
    ///     traits::zero::Zero,
    /// };
    ///
    /// fn main() {
    ///     let zc = Complex::<Double>::zero();
    ///     let c = Complex::of(Double::zero(), Double::of(0.0));
    ///     assert!((zc - c).is_zero());
    /// }
    /// ```
    fn zero() -> Self {
        Self {
            real: F::zero(),
            imaginary: F::zero(),
        }
    }

    /// A complex number is considered zero
    /// if and only if both its real and imaginary parts are regarded as zero.
    ///
    /// It is referred to as "regarded"
    /// because RealFloat can only be approximated to determine
    /// if they are zero due to errors.
    ///
    /// Commonly used to determine if two complex numbers are equal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double},
    ///     traits::zero::Zero,
    /// };
    ///
    /// fn main() {
    ///     let c1 = Complex::<Double>::of(Double::of(3.0), Double::of(4.0));
    ///     let c2 = Complex::of(Double::of(-3.0), Double::of(-4.0));
    ///     assert!((c1 + c2).is_zero());
    /// }
    /// ```
    fn is_zero(&self) -> bool { self.real.is_zero() && self.imaginary.is_zero() }
}

/// Implement the concept of ONE for the complex number.
impl<F: RealFloat> One for Complex<F> {
    /// Return the one complex number, which is 1 :+ 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double},
    ///     traits::{one::One, zero::Zero},
    /// };
    ///
    /// fn main() {
    ///     let one = Complex::<Double>::one();
    ///     let c = Complex::of(Double::of(1.0), Double::zero());
    ///     assert!((one - c).is_zero());
    /// }
    /// ```
    fn one() -> Self {
        Self {
            real: F::one(),
            imaginary: F::zero(),
        }
    }

    /// A complex number is considered one
    /// if and only if its real part is one and its imaginary part is zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double},
    ///     traits::{one::One, zero::Zero},
    /// };
    ///
    /// fn main() {
    ///     let one = Complex::<Double>::one();
    ///     let c1 = Complex::of(Double::of(3.0), Double::of(4.0));
    ///     let c2 = Complex::of(Double::of(3.0), Double::of(4.0));
    ///     assert!(((c1 / c2) - one).is_zero());
    /// }
    /// ```
    fn is_one(&self) -> bool { self.real.is_one() && self.imaginary.is_zero() }
}

/// Implement Default for the complex number.
impl<F: RealFloat> std::default::Default for Complex<F> {
    /// Return the default value of the complex number, which is ZERO.
    fn default() -> Self { Self::zero() }
}

/// Implement the negation operation for the complex number.
impl<F: RealFloat> std::ops::Neg for Complex<F> {
    type Output = Self;

    /// Perform the negation operation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double},
    ///     traits::zero::Zero,
    /// };
    ///
    /// fn main() {
    ///     let c1 = Complex::of(Double::of(3.0), Double::of(4.0));
    ///     let c2 = Complex::of(Double::of(-3.0), Double::of(-4.0));
    ///     assert!(((-c1) - c2).is_zero());
    /// }
    /// ```
    fn neg(self) -> Self::Output {
        Self {
            real: -self.real,
            imaginary: -self.imaginary,
        }
    }
}

/// Implement the addition operation for the complex number.
impl<F: RealFloat> std::ops::Add for Complex<F> {
    type Output = Self;

    /// Implement complex addition.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double},
    ///     traits::zero::Zero,
    /// };
    ///
    /// fn main() {
    ///     let c1 = Complex::of(Double::of(3.0), Double::of(4.0));
    ///     let c2 = Complex::of(Double::of(5.0), Double::of(-2.0));
    ///     let e = Complex::of(Double::of(8.0), Double::of(2.0));
    ///     assert!(((c1 + c2) - e).is_zero());
    /// }
    /// ```
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real + rhs.real,
            imaginary: self.imaginary + rhs.imaginary,
        }
    }
}

/// Implement the subtraction operation for the complex number.
impl<F: RealFloat> std::ops::Sub for Complex<F> {
    type Output = Self;

    /// Implement complex subtraction.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double},
    ///     traits::zero::Zero,
    /// };
    ///
    /// fn main() {
    ///     let c1 = Complex::of(Double::of(3.0), Double::of(4.0));
    ///     let c2 = Complex::of(Double::of(5.0), Double::of(-2.0));
    ///     let e = Complex::of(Double::of(-2.0), Double::of(6.0));
    ///     assert!(((c1 - c2) - e).is_zero());
    /// }
    /// ```
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real - rhs.real,
            imaginary: self.imaginary - rhs.imaginary,
        }
    }
}

/// Implement the multiplication operation for the complex number.
impl<F: RealFloat> std::ops::Mul for Complex<F> {
    type Output = Self;

    /// Implement complex multiplication.
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double},
    ///     traits::zero::Zero,
    /// };
    ///
    /// fn main() {
    ///     let c1 = Complex::of(Double::of(3.0), Double::of(4.0));
    ///     let c2 = Complex::of(Double::of(5.0), Double::of(-2.0));
    ///     let e = Complex::of(Double::of(23.0), Double::of(14.0));
    ///     assert!(((c1 * c2) - e).is_zero());
    /// }
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            real: self.real.clone() * rhs.real.clone()
                - self.imaginary.clone() * rhs.imaginary.clone(),
            imaginary: self.real * rhs.imaginary + self.imaginary * rhs.real,
        }
    }
}

/// Implement the division operation for the complex number.
impl<F: RealFloat> std::ops::Div for Complex<F> {
    type Output = Self;

    /// Implement complex division.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double},
    ///     traits::zero::Zero,
    /// };
    ///
    /// fn main() {
    ///     let c1 = Complex::of(Double::of(4.0), Double::of(2.0));
    ///     let c2 = Complex::of(Double::of(1.0), Double::of(-3.0));
    ///     let e = Complex::of(Double::of(-0.2), Double::of(1.4));
    ///     assert!(((c1 / c2) - e).is_zero());
    /// }
    /// ```
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

/// Implement concepts of NUMBER for complex numbers.
impl<F: RealFloat> Number for Complex<F> {
    /// The absolute value of a complex number is its norm,
    /// but the return type is a Complex instead of RealFloat.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double},
    ///     traits::{number::Number, zero::Zero},
    /// };
    ///
    /// fn main() {
    ///     let c = Complex::of(Double::of(4.0), Double::of(-3.0));
    ///     let c_abs = c.absolute_value();
    ///     assert!((c_abs.real - c.norm()).is_zero());
    ///     assert!(c_abs.imaginary.is_zero());
    /// }
    /// ```
    fn absolute_value(&self) -> Self {
        Self {
            real: self.clone().norm(),
            imaginary: F::zero(),
        }
    }

    /// Return the signed representation of a complex number.
    ///
    /// For the zero complex number, return the zero complex number.
    /// For other complex numbers, return the normalized complex number,
    /// which is obtained by dividing both the real and imaginary parts by its norm.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double},
    ///     traits::{number::Number, zero::Zero},
    /// };
    ///
    /// fn main() {
    ///     let z = Complex::<Double>::zero();
    ///     let c = Complex::of(Double::of(4.0), Double::of(-3.0));
    ///     let c_sign = Complex::of(Double::of(0.8), Double::of(-0.6));
    ///     assert!(z.sign_number().is_zero());
    ///     assert!((c.sign_number() - c_sign).is_zero());
    /// }
    /// ```
    fn sign_number(&self) -> Self {
        if self.is_zero() {
            Self::zero()
        } else {
            let n = self.clone().norm();
            Self {
                real: self.real.clone() / n.clone(),
                imaginary: self.imaginary.clone() / n,
            }
        }
    }

    /// Construct a complex number from an integer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double, integer::Integer},
    ///     traits::{number::Number, zero::Zero},
    /// };
    ///
    /// fn main() {
    ///     let i1 = Integer::of_str("12345678910").unwrap();
    ///     let c1 = Complex::<Double>::from_integer(i1);
    ///     assert!((c1 - Complex::of(Double::of(12345678910.0), Double::zero())).is_zero());
    /// }
    /// ```
    ///
    /// <div class="warning">
    ///
    /// If the size of the integer exceeds the maximum floating-point integer
    /// that can be represented by the corresponding RealFloat for a complex number,
    /// truncation or other issues may occur, resulting in significant errors.
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, double::Double, integer::Integer},
    ///     traits::{number::Number, zero::Zero},
    /// };
    ///
    /// fn main() {
    ///     let intnum = Integer::of_str("123456789101112131415161718192021222324252627").unwrap();
    ///     let c = Complex::<Double>::from_integer(intnum);
    ///     let real_c = Complex::<Double>::of(
    ///         Double::of(123456789101112130000000000000000000000000000.0),
    ///         Double::zero(),
    ///     );
    ///     assert!((c - real_c).is_zero());
    /// }
    /// ```
    ///
    /// </div>
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

/// Implement the concept of Floating for complex numbers.
impl<F: RealFloat> Floating for Complex<F> {
    const PI: Self = Self {
        real: F::PI,
        imaginary: F::ZERO,
    };
    const ZERO: Self = Self {
        real: F::ZERO,
        imaginary: F::ZERO,
    };

    fn exponential(self) -> Self {
        Self {
            real: self.real.clone().exponential() * self.imaginary.clone().cosine(),
            imaginary: self.real.exponential() * self.imaginary.sine(),
        }
    }

    fn logarithmic(self) -> Self {
        Self {
            real: self.clone().norm().logarithmic(),
            imaginary: F::arc_tangent_2(self.imaginary, self.real),
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
            * (i * self.clone() + (Self::one() - self.clone() * self).square_root())
                .logarithmic()
    }

    fn arc_cosine(self) -> Self {
        let i = Self {
            real: F::zero(),
            imaginary: F::one(),
        };
        -i.clone()
            * (self.clone() + i * (Self::one() - self.clone() * self).square_root())
                .logarithmic()
    }

    fn arc_tangent(self) -> Self {
        let i = Self {
            real: F::zero(),
            imaginary: F::one(),
        };
        let two = Self::one() + Self::one();
        Self::one() / (two * i.clone())
            * ((Self::one() + i.clone() * self.clone()) / (Self::one() - i * self))
                .logarithmic()
    }

    fn hyperbolic_sine(self) -> Self {
        (self.clone().exponential() - (-self).exponential()) / (Self::one() + Self::one())
    }

    fn hyperbolic_cosine(self) -> Self {
        (self.clone().exponential() + (-self).exponential()) / (Self::one() + Self::one())
    }

    fn arc_hyperbolic_sine(self) -> Self {
        (self.clone() + (Self::one() + self.clone() * self).square_root()).logarithmic()
    }

    fn arc_hyperbolic_cosine(self) -> Self {
        (self.clone() + (self.clone() * self - Self::one()).square_root()).logarithmic()
    }

    fn arc_hyperbolic_tangent(self) -> Self {
        Self::half() * ((Self::one() + self.clone()) / (Self::one() - self)).logarithmic()
    }
}

/// Implement the concept of Fractional for complex numbers.
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

/// Implement PartialOrd for complex numbers with zero imaginary part.
impl<F: RealFloat> std::cmp::PartialOrd for Complex<F> {
    /// Only complex numbers with zero imaginary part can be compared,
    /// otherwise, return None.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{complex::Complex, float::Float},
    ///     traits::zero::Zero,
    /// };
    ///
    /// fn main() {
    ///     let c1 = Complex::of(Float::of(3.0), Float::zero());
    ///     let c2 = Complex::of(Float::of(-2.0), Float::zero());
    ///     let c3 = Complex::of(Float::of(2.0), Float::of(2.0));
    ///
    ///     assert_eq!(c1.partial_cmp(&c2), Some(std::cmp::Ordering::Greater));
    ///     assert_eq!(c1.partial_cmp(&c3), None);
    /// }
    /// ```
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.imaginary.is_zero() && other.imaginary.is_zero() {
            self.real.partial_cmp(&other.real)
        } else {
            None
        }
    }
}

/// Implement Display for complex numbers.
impl<F: RealFloat> std::fmt::Display for Complex<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} :+ {}", self.real, self.imaginary)
    }
}

/// Implement Debug for complex numbers.
impl<F: RealFloat> std::fmt::Debug for Complex<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+} :+ {:+}", self.real, self.imaginary)
    }
}

/// Implement FromStr for complex numbers.
impl<F: RealFloat> std::str::FromStr for Complex<F> {
    type Err = ();

    /// Construct a complex number from a string.
    ///
    /// A standard complex literal is `A :+ B`,
    /// where both `A` and `B` do not include parentheses,
    /// for example, `1.0 :+ -3.0`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.split(":+").map(|p| p.trim()).collect::<Vec<_>>();
        if trimmed_s.len() != 2 {
            eprintln!(
                "Error[Complex::from_str]: ({}) is not a valid Complex literal.",
                s
            );
            Err(())
        } else {
            let real = F::from_str(trimmed_s[0])?;
            let imaginary = F::from_str(trimmed_s[1])?;
            Ok(Self { real, imaginary })
        }
    }
}

/// Uniform distribution of complex numbers.
pub struct UniformComplex<F: RealFloat + SampleUniform>(Uniform<F>, Uniform<F>);

/// Implement uniform sampling for the uniform distribution of complex numbers.
impl<F: RealFloat + SampleUniform> UniformSampler for UniformComplex<F> {
    type X = Complex<F>;

    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(
            Uniform::<F>::new(low.borrow().real.clone(), high.borrow().real.clone()),
            Uniform::<F>::new(
                low.borrow().imaginary.clone(),
                high.borrow().imaginary.clone(),
            ),
        )
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(
            Uniform::<F>::new_inclusive(low.borrow().real.clone(), high.borrow().real.clone()),
            Uniform::<F>::new_inclusive(
                low.borrow().imaginary.clone(),
                high.borrow().imaginary.clone(),
            ),
        )
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        Self::X::of(self.0.sample(rng), self.1.sample(rng))
    }
}

/// Implement uniform sampling for complex numbers.
impl<F: RealFloat + SampleUniform> SampleUniform for Complex<F> {
    type Sampler = UniformComplex<F>;
}
