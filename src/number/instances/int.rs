//! # Number Type :: Int
//!
//! i32 wrapper.

use crate::number::{
    instances::{integer::Integer, ratio::Rational},
    traits::{integral::Integral, number::Number, one::One, real::Real, zero::Zero},
};

#[cfg(feature = "rand_mat")]
use rand::{
    distributions::uniform::{SampleBorrow, SampleUniform, UniformInt, UniformSampler},
    Rng,
};

/// Int
#[derive(Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_mat", derive(serde::Deserialize, serde::Serialize))]
pub struct Int {
    inner: i32,
}

impl Int {
    pub const fn of(num: i32) -> Self {
        Self { inner: num }
    }

    // create Int from &str
    pub fn of_str(int_number: &str) -> Result<Self, String> {
        std::str::FromStr::from_str(int_number)
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

impl Zero for Int {
    fn zero() -> Self {
        Self { inner: 0i32 }
    }

    fn is_zero(&self) -> bool {
        self.inner == 0i32
    }
}

impl One for Int {
    fn one() -> Self {
        Self { inner: 1i32 }
    }

    fn is_one(&self) -> bool {
        self.inner == 1i32
    }
}

impl std::default::Default for Int {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::ops::Neg for Int {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self { inner: -self.inner }
    }
}

impl std::ops::Add for Int {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

impl std::ops::Sub for Int {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner - rhs.inner,
        }
    }
}

impl std::ops::Mul for Int {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner * rhs.inner,
        }
    }
}

impl Number for Int {
    fn absolute_value(&self) -> Self {
        Self {
            inner: self.inner.abs(),
        }
    }

    fn sign_number(&self) -> Self {
        if self.inner == 0i32 {
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
            let inner = format!("{}", integer_number)
                .parse::<i32>()
                .expect("Error[Int::from<Integer>]: integer_number should be a proper i32 number");
            Self { inner }
        }
    }
}

impl Real for Int {
    fn to_rational(self) -> Rational
    where
        Self: crate::number::traits::integral::Integral,
    {
        Rational::of(self.to_integer(), Integer::one())
    }
}

impl Integral for Int {
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

impl std::fmt::Display for Int {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl std::fmt::Debug for Int {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+}", self.inner)
    }
}

impl std::str::FromStr for Int {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.trim();
        if let Ok(num) = trimmed_s.parse::<i32>() {
            Ok(Self { inner: num })
        } else {
            Err(format!(
                "Error[Int::from_str]: {} is not a valid integer",
                trimmed_s
            ))
        }
    }
}

#[cfg(feature = "rand_mat")]
#[doc(cfg(feature = "rand_mat"))]
/// Uniform for Int
pub struct UniformI32(UniformInt<i32>);

#[cfg(feature = "rand_mat")]
impl UniformSampler for UniformI32 {
    type X = Int;

    fn new<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(UniformInt::<i32>::new(
            low.borrow().inner,
            high.borrow().inner,
        ))
    }

    fn new_inclusive<B1, B2>(low: B1, high: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Self(UniformInt::<i32>::new_inclusive(
            low.borrow().inner,
            high.borrow().inner,
        ))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        Self::X::of(self.0.sample(rng))
    }
}

#[cfg(feature = "rand_mat")]
impl SampleUniform for Int {
    type Sampler = UniformI32;
}
