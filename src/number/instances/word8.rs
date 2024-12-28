//! # instances::word8
//!
//! Functions and implementations related to 8-bit unsigned integers.

use rand::{
    Rng,
    distributions::uniform::{SampleBorrow, SampleUniform, UniformInt, UniformSampler},
};

use crate::number::{
    instances::{integer::Integer, ratio::Rational},
    traits::{integral::Integral, number::Number, one::One, real::Real, zero::Zero},
};

/// Word8 numbers are the wrapper type for u8.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Word8 {
    inner: u8,
}

impl Word8 {
    /// Construct Word8 numbers from u8.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::word8::Word8;
    ///
    /// fn main() { let _w = Word8::of(12); }
    /// ```
    pub const fn of(num: u8) -> Self { Self { inner: num } }

    /// Construct word8 numbers from string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::word8::Word8;
    ///
    /// fn main() {
    ///     let sw = Word8::of_str("23").unwrap();
    ///     let w = Word8::of(23);
    ///     assert_eq!(sw, w);
    /// }
    /// ```
    pub fn of_str(uint8_number: &str) -> Option<Self> {
        std::str::FromStr::from_str(uint8_number).ok()
    }

    /// Return the digit at each position.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::word8::Word8;
    ///
    /// fn main() {
    ///     let w = Word8::of(254);
    ///     let digits = w.digits();
    ///     assert_eq!(digits, vec![2, 5, 4]);
    /// }
    /// ```
    pub fn digits(&self) -> Vec<u8> {
        let string_view = self.inner.to_string();
        string_view
            .chars()
            .map(|digit: char| digit as u8 - '0' as u8)
            .collect::<Vec<_>>()
    }
}

/// Implement the concept of ZERO for the word8 number.
impl Zero for Word8 {
    fn zero() -> Self { Self { inner: 0u8 } }

    fn is_zero(&self) -> bool { self.inner == 0u8 }
}

/// Implement the concept of ONE for the word8 number.
impl One for Word8 {
    fn one() -> Self { Self { inner: 1u8 } }

    fn is_one(&self) -> bool { self.inner == 1u8 }
}

/// Implement Default for the word8 number.
impl std::default::Default for Word8 {
    fn default() -> Self { Self::zero() }
}

/// Implement the negation operation for the word8 number.
impl std::ops::Neg for Word8 {
    type Output = Self;

    /// Retrieve the corresponding opposite number.
    ///
    /// For an unsigned number `A`, we have its opposite number `B`.
    /// By the definition of the opposite number, we know that `A + B = 0`,
    /// which means B = MAX - A.
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::word8::Word8;
    ///
    /// fn main() {
    ///     let w = Word8::of(12);
    ///     assert_eq!(-w, Word8::of(243));
    /// }
    /// ```
    fn neg(self) -> Self::Output {
        Self {
            inner: u8::MAX - self.inner,
        }
    }
}

/// Implement the addition operation for the word8 number.
impl std::ops::Add for Word8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

/// Implement the subtraction operation for the word8 number.
impl std::ops::Sub for Word8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: if self.inner > rhs.inner {
                self.inner - rhs.inner
            } else {
                u8::MAX - rhs.inner + self.inner
            },
        }
    }
}

/// Implement the multiplication operation for the word8 number.
impl std::ops::Mul for Word8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner * rhs.inner,
        }
    }
}

/// Implement the concept of NUMBER for the word8 number.
impl Number for Word8 {
    fn absolute_value(&self) -> Self { Self { inner: self.inner } }

    fn sign_number(&self) -> Self {
        if self.inner == 0u8 {
            Self::zero()
        } else {
            Self::one()
        }
    }

    /// Construct a word8 number from an integer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{integer::Integer, word8::Word8},
    ///     traits::number::Number,
    /// };
    ///
    /// fn main() {
    ///     let integer = Integer::of_str("18");
    ///     let word = integer.map(|w| Word8::from_integer(w));
    ///     assert_eq!(word, Some(Word8::of(18)));
    /// }
    /// ```
    ///
    /// ## Panics
    ///
    /// If the size of the integer exceeds the maximum literal value of u8,
    /// it will cause a panic.
    ///
    /// ```rust,should_panic
    /// use rmatrix_ks::number::{
    ///     instances::{integer::Integer, word8::Word8},
    ///     traits::number::Number,
    /// };
    ///
    /// fn main() {
    ///     let integer = Integer::of_str("576");
    ///     // Panic occurs here.
    ///     let _ = integer.map(|w| Word8::from_integer(w));
    /// }
    /// ```
    fn from_integer(integer_number: Integer) -> Self {
        if integer_number.is_zero() {
            Self::zero()
        } else {
            let inner = format!("{:?}", integer_number)
                .parse::<u8>()
                .expect(&format!(
                    "Error[Word8::from_Integer]: ({}) should be a valid u8 number.",
                    integer_number
                ));
            Self { inner }
        }
    }
}

/// Implement the concept of Real for Word8.
impl Real for Word8 {
    fn to_rational(self) -> Rational
    where
        Self: crate::number::traits::integral::Integral,
    {
        Rational::of(self.to_integer(), Integer::one())
    }
}

/// Implement the concept of Integral for Word8.
impl Integral for Word8 {
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
        Integer::of_str(&format!("{}", self)).expect(&format!(
            "Error[Word8::to_integer]: ({}) should be a valid Integer.",
            self
        ))
    }
}

/// Implement Display for Word8.
impl std::fmt::Display for Word8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

/// Implement Debug for Word8.
impl std::fmt::Debug for Word8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+}", self.inner)
    }
}

/// Implement FromStr for Word8.
impl std::str::FromStr for Word8 {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.trim();
        if let Ok(num) = trimmed_s.parse::<u8>() {
            Ok(Self { inner: num })
        } else {
            eprintln!(
                "Error[Word8::from_str]: ({}) is not a valid Word8 literal.",
                trimmed_s
            );
            Err(())
        }
    }
}

/// Uniform distribution of word8 numbers.
pub struct UniformU8(UniformInt<u8>);

/// Implement uniform sampling for the uniform distribution of word8 numbers.
impl UniformSampler for UniformU8 {
    type X = Word8;

    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(UniformInt::<u8>::new(
            low.borrow().inner,
            high.borrow().inner,
        ))
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(UniformInt::<u8>::new_inclusive(
            low.borrow().inner,
            high.borrow().inner,
        ))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        Self::X::of(self.0.sample(rng))
    }
}

/// Implement uniform sampling for word8 numbers.
impl SampleUniform for Word8 {
    type Sampler = UniformU8;
}
