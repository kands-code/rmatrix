//! # Number Type :: Word8
//!
//! u8 wrapper.

use crate::number::{
    instances::{integer::Integer, ratio::Rational},
    traits::{integeral::Integral, number::Number, one::One, real::Real, zero::Zero},
};

/// Word8
#[derive(Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_mat", derive(serde::Deserialize, serde::Serialize))]
pub struct Word8 {
    inner: u8,
}

impl Word8 {
    pub const fn of(num: u8) -> Self {
        Self { inner: num }
    }

    // create Word8 from &str
    pub fn of_str(uint8_number: &str) -> Result<Self, String> {
        std::str::FromStr::from_str(uint8_number)
    }

    pub fn digits(&self) -> Vec<u8> {
        let string_view = self.inner.to_string();
        string_view
            .chars()
            .map(|digit: char| digit as u8 - '0' as u8)
            .collect::<Vec<_>>()
    }
}

impl Zero for Word8 {
    fn zero() -> Self {
        Self { inner: 0u8 }
    }

    fn is_zero(&self) -> bool {
        self.inner == 0u8
    }
}

impl One for Word8 {
    fn one() -> Self {
        Self { inner: 1u8 }
    }

    fn is_one(&self) -> bool {
        self.inner == 1u8
    }
}

impl std::ops::Neg for Word8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            inner: u8::MAX - self.inner,
        }
    }
}

impl std::ops::Add for Word8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

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

impl std::ops::Mul for Word8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner * rhs.inner,
        }
    }
}

impl Number for Word8 {
    fn absolute_value(&self) -> Self {
        Self { inner: self.inner }
    }

    fn sign_number(&self) -> Self {
        if self.inner == 0u8 {
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
                .parse::<u8>()
                .expect("Error[Word8::from<Integer>]: integer_number should be a proper u8 number");
            Self { inner }
        }
    }
}

impl Real for Word8 {
    fn to_rational(self) -> Rational
    where
        Self: crate::number::traits::integeral::Integral,
    {
        Rational::of(self.to_integer(), Integer::one())
    }
}

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
        let sign = !(self < Self::zero());
        Integer::of(sign, &self.digits()).expect("")
    }
}

impl std::fmt::Display for Word8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl std::fmt::Debug for Word8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+}", self.inner)
    }
}

impl std::str::FromStr for Word8 {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.trim();
        if let Ok(num) = trimmed_s.parse::<u8>() {
            Ok(Self { inner: num })
        } else {
            Err(format!(
                "Error[Word8::from_str]: {} is not a valid integer",
                trimmed_s
            ))
        }
    }
}
