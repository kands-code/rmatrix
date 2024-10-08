//! # Number Type :: Int8
//!
//! i8 wrapper.

use crate::number::instances::integer::Integer;
use crate::number::instances::ratio::Rational;
use crate::number::traits::integeral::Integral;
use crate::number::traits::number::Number;
use crate::number::traits::one::One;
use crate::number::traits::real::Real;
use crate::number::traits::zero::Zero;

/// Int8
#[derive(Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_mat", derive(serde::Deserialize, serde::Serialize))]
pub struct Int8 {
    inner: i8,
}

impl Int8 {
    pub const fn of(num: i8) -> Self {
        Self { inner: num }
    }

    // create Int8 from &str
    pub fn of_str(int8_number: &str) -> Result<Self, String> {
        std::str::FromStr::from_str(int8_number)
    }

    pub fn digits(&self) -> Vec<u8> {
        let string_view = self.inner.to_string();
        string_view
            .chars()
            .map(|digit: char| digit as u8 - '0' as u8)
            .collect::<Vec<_>>()
    }
}

impl Zero for Int8 {
    fn zero() -> Self {
        Self { inner: 0i8 }
    }

    fn is_zero(&self) -> bool {
        self.inner == 0i8
    }
}

impl One for Int8 {
    fn one() -> Self {
        Self { inner: 1i8 }
    }

    fn is_one(&self) -> bool {
        self.inner == 1i8
    }
}

impl std::ops::Neg for Int8 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { inner: -self.inner }
    }
}

impl std::ops::Add for Int8 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

impl std::ops::Sub for Int8 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner - rhs.inner,
        }
    }
}

impl std::ops::Mul for Int8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner * rhs.inner,
        }
    }
}

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

    fn from_integer(integer_number: Integer) -> Self {
        if integer_number.is_zero() {
            Self::zero()
        } else {
            let integer_value = integer_number
                .digits()
                .iter()
                .map(|&v| (v + '0' as u8) as char)
                .collect::<String>()
                .parse::<i8>().expect("Error[Int8::from<Int8eger>]: Int8eger::digits() should be a reasonable numeric string");
            Self {
                inner: integer_value * if integer_number.sign { 1i8 } else { -1i8 },
            }
        }
    }
}

impl Real for Int8 {
    fn to_rational(self) -> Rational
    where
        Self: crate::number::traits::integeral::Integral,
    {
        Rational::of(self.to_integer(), Integer::one())
    }
}

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
        let sign = !(self < Self::zero());
        Integer::of(sign, &self.digits()).expect("")
    }
}

impl std::fmt::Display for Int8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl std::fmt::Debug for Int8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+}", self.inner)
    }
}

impl std::str::FromStr for Int8 {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.trim();
        if let Ok(num) = trimmed_s.parse::<i8>() {
            Ok(Self { inner: num })
        } else {
            Err(format!(
                "Error[Int8::from_str]: {} is not a valid integer",
                trimmed_s
            ))
        }
    }
}
