//! # instances::int8
//!
//! Functions and implementations related to 8-bit signed integers.

use rand::{
    Rng,
    distr::uniform::{SampleBorrow, SampleUniform, UniformInt, UniformSampler},
};

use crate::number::{
    instances::{integer::Integer, ratio::Rational},
    traits::{integral::Integral, number::Number, one::One, real::Real, zero::Zero},
};

/// Int8 numbers are the wrapper type for i8.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Int8 {
    inner: i8,
}

impl Int8 {
    /// Construct int8 numbers from i8.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::int8::Int8;
    ///
    /// let _i = Int8::of(12);
    /// ```
    pub const fn of(num: i8) -> Self { Self { inner: num } }

    /// Construct int8 numbers from string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::int8::Int8;
    ///
    /// let si = Int8::of_str("-23").unwrap();
    /// let i = Int8::of(-23);
    /// assert_eq!(si, i);
    /// ```
    pub fn of_str(int8_number: &str) -> Option<Self> { std::str::FromStr::from_str(int8_number).ok() }

    /// Return the digit at each position.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::int8::Int8;
    ///
    /// let i = Int8::of(125);
    /// let digits = i.digits();
    /// assert_eq!(digits, vec![1, 2, 5]);
    /// ```
    pub fn digits(&self) -> Vec<u8> {
        let string_view = self.inner.to_string();
        string_view
            .chars()
            .map(|digit: char| digit as u8 - b'0')
            .collect::<Vec<_>>()
    }

    /// Get the raw value of a int8 number.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::int8::Int8;
    ///
    /// let i = Int8::of(69);
    /// assert_eq!(i.raw(), 69);
    /// ```
    pub fn raw(&self) -> i8 { self.inner }
}

/// Implement the concept of ZERO for the int8 number.
impl Zero for Int8 {
    fn zero() -> Self { Self { inner: 0i8 } }

    fn is_zero(&self) -> bool { self.inner == 0i8 }
}

/// Implement the concept of ONE for the int8 number.
impl One for Int8 {
    fn one() -> Self { Self { inner: 1i8 } }

    fn is_one(&self) -> bool { self.inner == 1i8 }
}

/// Implement Default for the int8 number.
impl std::default::Default for Int8 {
    fn default() -> Self { Self::zero() }
}

/// Implement the negation operation for the int8 number.
impl std::ops::Neg for Int8 {
    type Output = Self;

    fn neg(self) -> Self::Output { Self { inner: -self.inner } }
}

/// Implement the addition operation for the int8 number.
impl std::ops::Add for Int8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

/// Implement the subtraction operation for the int8 number.
impl std::ops::Sub for Int8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner - rhs.inner,
        }
    }
}

/// Implement the multiplication operation for the int8 number.
impl std::ops::Mul for Int8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner * rhs.inner,
        }
    }
}

/// Implement the concept of NUMBER for the int8 number.
impl Number for Int8 {
    fn absolute_value(&self) -> Self {
        Self {
            inner: self.inner.abs(),
        }
    }

    fn sign_number(&self) -> Self {
        if self.inner == 0i8 {
            Self::zero()
        } else if self.inner > 0 {
            Self::one()
        } else {
            -Self::one()
        }
    }

    /// Construct a int8 number from an integer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{int8::Int8, integer::Integer},
    ///     traits::number::Number,
    /// };
    ///
    /// let integer = Integer::of_str("-18");
    /// let int = integer.map(|i| Int8::from_integer(i));
    /// assert_eq!(int, Some(Int8::of(-18)));
    /// ```
    ///
    /// ## Panics
    ///
    /// If the size of the integer exceeds the maximum literal value of i8,
    /// it will cause a panic.
    ///
    /// ```rust,should_panic
    /// use rmatrix_ks::number::{
    ///     instances::{int8::Int8, integer::Integer},
    ///     traits::number::Number,
    /// };
    ///
    /// let integer = Integer::of_str("12345678");
    /// // Panic occurs here.
    /// let _ = integer.map(|i| Int8::from_integer(i));
    /// ```
    fn from_integer(integer_number: Integer) -> Self {
        if integer_number.is_zero() {
            Self::zero()
        } else {
            let inner = format!("{integer_number:?}")
                .parse::<i8>()
                .unwrap_or_else(|_| panic!("Error[Int8::from_Integer]: ({integer_number}) should be a valid i8 number."));
            Self { inner }
        }
    }
}

/// Implement the concept of Real for Int8.
impl Real for Int8 {
    fn to_rational(self) -> Rational
    where
        Self: crate::number::traits::integral::Integral,
    {
        Rational::of(self.to_integer(), Integer::one())
    }
}

/// Implement the concept of Integral for Int8.
impl Integral for Int8 {
    fn quot_rem(self, rhs: Self) -> (Self, Self) {
        (
            Self {
                inner: self.inner / rhs.inner,
            },
            Self {
                inner: self.inner % rhs.inner,
            },
        )
    }

    fn div_mod(self, rhs: Self) -> (Self, Self) {
        let (quot, rem) = self.clone().quot_rem(rhs.clone());
        if rem.is_zero() || (quot >= Self::zero() && rem > Self::zero()) {
            (quot, rem)
        } else {
            let div = quot - Self::one();
            (div.clone(), self - div * rhs)
        }
    }

    fn to_integer(self) -> Integer {
        Integer::of_str(&format!("{self}"))
            .unwrap_or_else(|| panic!("Error[Int8::to_integer]: ({self}) should be a valid Integer."))
    }
}

/// Implement Display for Int8.
impl std::fmt::Display for Int8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.inner) }
}

/// Implement Debug for Int8.
impl std::fmt::Debug for Int8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{:+}", self.inner) }
}

/// Implement FromStr for Int8.
impl std::str::FromStr for Int8 {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.trim();
        if let Ok(num) = trimmed_s.parse::<i8>() {
            Ok(Self { inner: num })
        } else {
            eprintln!("Error[Int8::from_str]: ({trimmed_s}) is not a valid Int8 literal.");
            Err(())
        }
    }
}

/// Uniform distribution of int8 numbers.
pub struct UniformI8(UniformInt<i8>);

/// Implement uniform sampling for the uniform distribution of int8 numbers.
impl UniformSampler for UniformI8 {
    type X = Int8;

    fn new<B1, B2>(low: B1, high: B2) -> Result<UniformI8, rand::distr::uniform::Error>
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Ok(Self(UniformInt::<i8>::new(
            low.borrow().inner,
            high.borrow().inner,
        )?))
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Result<UniformI8, rand::distr::uniform::Error>
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Ok(Self(UniformInt::<i8>::new_inclusive(
            low.borrow().inner,
            high.borrow().inner,
        )?))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X { Self::X::of(self.0.sample(rng)) }
}

/// Implement uniform sampling for int8 numbers.
impl SampleUniform for Int8 {
    type Sampler = UniformI8;
}
