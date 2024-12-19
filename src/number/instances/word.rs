//! # Number Type :: Word
//!
//! u32 wrapper.

use crate::number::{
    instances::{integer::Integer, ratio::Rational},
    traits::{integral::Integral, number::Number, one::One, real::Real, zero::Zero},
};

use rand::{
    distributions::uniform::{SampleBorrow, SampleUniform, UniformInt, UniformSampler},
    Rng,
};

/// Word
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Word {
    inner: u32,
}

impl Word {
    pub const fn of(num: u32) -> Self {
        Self { inner: num }
    }

    // create Word from &str
    pub fn of_str(int_number: &str) -> Option<Self> {
        std::str::FromStr::from_str(int_number).ok()
    }

    pub fn digits(&self) -> Vec<u8> {
        let string_view = self.inner.to_string();
        string_view
            .chars()
            .skip_while(|ch| !ch.is_ascii_digit())
            .map(|digit: char| digit as u8 - '0' as u8)
            .collect::<Vec<_>>()
    }
}

impl Zero for Word {
    fn zero() -> Self {
        Self { inner: 0u32 }
    }

    fn is_zero(&self) -> bool {
        self.inner == 0u32
    }
}

impl One for Word {
    fn one() -> Self {
        Self { inner: 1u32 }
    }

    fn is_one(&self) -> bool {
        self.inner == 1u32
    }
}

impl std::default::Default for Word {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::ops::Neg for Word {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            inner: u32::MAX - self.inner,
        }
    }
}

impl std::ops::Add for Word {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

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

impl std::ops::Mul for Word {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner * rhs.inner,
        }
    }
}

impl Number for Word {
    fn absolute_value(&self) -> Self {
        Self { inner: self.inner }
    }

    fn sign_number(&self) -> Self {
        if self.inner == 0u32 {
            Self::zero()
        } else {
            Self::one()
        }
    }

    fn from_integer(integer_number: Integer) -> Self {
        if integer_number.is_zero() {
            Self::zero()
        } else {
            let inner = format!("{}", integer_number)
                .parse::<u32>()
                .expect(&format!(
                    "Error[Word::from_Integer]: ({}) should be a valid u32 number.",
                    integer_number
                ));
            Self { inner }
        }
    }
}

impl Real for Word {
    fn to_rational(self) -> Rational
    where
        Self: crate::number::traits::integral::Integral,
    {
        Rational::of(self.to_integer(), Integer::one())
    }
}

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
        Integer::of_str(&format!("{}", self)).expect(&format!(
            "Error[Word::to_integer]: ({}) should be a valid Integer.",
            self
        ))
    }
}

impl std::fmt::Display for Word {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl std::fmt::Debug for Word {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+}", self.inner)
    }
}

impl std::str::FromStr for Word {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.trim();
        if let Ok(num) = trimmed_s.parse::<u32>() {
            Ok(Self { inner: num })
        } else {
            eprintln!(
                "Error[Word::from_str]: ({}) is not a valid Word literal.",
                trimmed_s
            );
            Err(())
        }
    }
}

/// Uniform for Word
pub struct UniformU32(UniformInt<u32>);

impl UniformSampler for UniformU32 {
    type X = Word;

    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(UniformInt::<u32>::new(
            low.borrow().inner,
            high.borrow().inner,
        ))
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(UniformInt::<u32>::new_inclusive(
            low.borrow().inner,
            high.borrow().inner,
        ))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        Self::X::of(self.0.sample(rng))
    }
}

impl SampleUniform for Word {
    type Sampler = UniformU32;
}
