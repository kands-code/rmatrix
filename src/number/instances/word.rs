//! # instances::word
//!
//! Functions and implementations related to 32-bit unsigned integers.

use rand::{
    Rng,
    distr::uniform::{SampleBorrow, SampleUniform, UniformInt, UniformSampler},
};

use crate::number::{
    instances::{integer::Integer, ratio::Rational},
    traits::{integral::Integral, number::Number, one::One, real::Real, zero::Zero},
};

/// Word numbers are the wrapper type for u32.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Word {
    inner: u32,
}

impl Word {
    /// Construct word numbers from u32.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::word::Word;
    ///
    /// let _w = Word::of(12);
    /// ```
    pub const fn of(num: u32) -> Self { Self { inner: num } }

    /// Construct word numbers from string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::word::Word;
    ///
    /// let sw = Word::of_str("23").unwrap();
    /// let w = Word::of(23);
    /// assert_eq!(sw, w);
    /// ```
    pub fn of_str(int_number: &str) -> Option<Self> { std::str::FromStr::from_str(int_number).ok() }

    /// Return the digit at each position.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::word::Word;
    ///
    /// let w = Word::of(125436);
    /// let digits = w.digits();
    /// assert_eq!(digits, vec![1, 2, 5, 4, 3, 6]);
    /// ```
    pub fn digits(&self) -> Vec<u8> {
        let string_view = self.inner.to_string();
        string_view
            .chars()
            .skip_while(|ch| !ch.is_ascii_digit())
            .map(|digit: char| digit as u8 - b'0')
            .collect::<Vec<_>>()
    }

    /// Get the raw value of a word number.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::word::Word;
    ///
    /// let w = Word::of(69);
    /// assert_eq!(w.raw(), 69);
    /// ```
    pub fn raw(&self) -> u32 { self.inner }
}

/// Implement the concept of ZERO for the word number.
impl Zero for Word {
    fn zero() -> Self { Self { inner: 0u32 } }

    fn is_zero(&self) -> bool { self.inner == 0u32 }
}

/// Implement the concept of ONE for the word number.
impl One for Word {
    fn one() -> Self { Self { inner: 1u32 } }

    fn is_one(&self) -> bool { self.inner == 1u32 }
}

/// Implement Default for the word number.
impl std::default::Default for Word {
    fn default() -> Self { Self::zero() }
}

/// Implement the negation operation for the word number.
impl std::ops::Neg for Word {
    type Output = Self;

    /// Retrieve the corresponding opposite number.
    ///
    /// For an unsigned number `A`, we have its opposite number `B`.
    /// By the definition of the opposite number, we know that `A + B = 0`,
    /// which means B = MAX - A.
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::word::Word;
    ///
    /// let w = Word::of(224756);
    /// assert_eq!(-w, Word::of(4294742539));
    /// ```
    fn neg(self) -> Self::Output {
        Self {
            inner: u32::MAX - self.inner,
        }
    }
}

/// Implement the addition operation for the word number.
impl std::ops::Add for Word {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

/// Implement the subtraction operation for the word number.
impl std::ops::Sub for Word {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: if self.inner > rhs.inner {
                self.inner - rhs.inner
            } else {
                u32::MAX - rhs.inner + self.inner
            },
        }
    }
}

/// Implement the multiplication operation for the word number.
impl std::ops::Mul for Word {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner * rhs.inner,
        }
    }
}

/// Implement the concept of NUMBER for the word number.
impl Number for Word {
    fn absolute_value(&self) -> Self { Self { inner: self.inner } }

    fn sign_number(&self) -> Self {
        if self.inner == 0u32 {
            Self::zero()
        } else {
            Self::one()
        }
    }

    /// Construct a word number from an integer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{integer::Integer, word::Word},
    ///     traits::number::Number,
    /// };
    ///
    /// let digits = (1..10).collect::<Vec<u8>>();
    /// let integer = Integer::of(true, &digits);
    /// let word = integer.map(|w| Word::from_integer(w));
    /// assert_eq!(word, Some(Word::of(123456789)));
    /// ```
    ///
    /// ## Panics
    ///
    /// If the size of the integer exceeds the maximum literal value of u32,
    /// it will cause a panic.
    ///
    /// ```rust,should_panic
    /// use rmatrix_ks::number::{
    ///     instances::{integer::Integer, word::Word},
    ///     traits::number::Number,
    /// };
    ///
    /// let integer = Integer::of_str("12345678910111213141516");
    /// // Panic occurs here.
    /// let _ = integer.map(|w| Word::from_integer(w));
    /// ```
    fn from_integer(integer_number: Integer) -> Self {
        if integer_number.is_zero() {
            Self::zero()
        } else {
            let inner = format!("{integer_number:?}")
                .parse::<u32>()
                .unwrap_or_else(|_| panic!("Error[Word::from_Integer]: ({integer_number}) should be a valid u32 number."));
            Self { inner }
        }
    }
}

/// Implement the concept of Real for Word.
impl Real for Word {
    fn to_rational(self) -> Rational
    where
        Self: crate::number::traits::integral::Integral,
    {
        Rational::of(self.to_integer(), Integer::one())
    }
}

/// Implement the concept of Integral for Word.
impl Integral for Word {
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
            .unwrap_or_else(|| panic!("Error[Word::to_integer]: ({self}) should be a valid Integer."))
    }
}

/// Implement Display for Word.
impl std::fmt::Display for Word {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.inner) }
}

/// Implement Debug for Word.
impl std::fmt::Debug for Word {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{:+}", self.inner) }
}

/// Implement FromStr for Word.
impl std::str::FromStr for Word {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.trim();
        if let Ok(num) = trimmed_s.parse::<u32>() {
            Ok(Self { inner: num })
        } else {
            eprintln!("Error[Word::from_str]: ({trimmed_s}) is not a valid Word literal.");
            Err(())
        }
    }
}

/// Uniform distribution of word numbers.
pub struct UniformU32(UniformInt<u32>);

/// Implement uniform sampling for the uniform distribution of word numbers.
impl UniformSampler for UniformU32 {
    type X = Word;

    fn new<B1, B2>(low: B1, high: B2) -> Result<UniformU32, rand::distr::uniform::Error>
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Ok(Self(UniformInt::<u32>::new(
            low.borrow().inner,
            high.borrow().inner,
        )?))
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Result<UniformU32, rand::distr::uniform::Error>
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Ok(Self(UniformInt::<u32>::new_inclusive(
            low.borrow().inner,
            high.borrow().inner,
        )?))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X { Self::X::of(self.0.sample(rng)) }
}

/// Implement uniform sampling for word numbers.
impl SampleUniform for Word {
    type Sampler = UniformU32;
}
