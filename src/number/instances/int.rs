//! # instances::int
//!
//! Functions and implementations related to 32-bit signed integers.

use rand::{
    Rng,
    distr::uniform::{SampleBorrow, SampleUniform, UniformInt, UniformSampler},
};

use crate::number::{
    instances::{integer::Integer, ratio::Rational},
    traits::{integral::Integral, number::Number, one::One, real::Real, zero::Zero},
};

/// Int numbers are the wrapper type for i32.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Int {
    inner: i32,
}

impl Int {
    /// Construct int numbers from i32.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::int::Int;
    ///
    /// fn main() { let _i = Int::of(12); }
    /// ```
    pub const fn of(num: i32) -> Self { Self { inner: num } }

    /// Construct int numbers from string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::int::Int;
    ///
    /// fn main() {
    ///     let si = Int::of_str("23").unwrap();
    ///     let i = Int::of(23);
    ///     assert_eq!(si, i);
    /// }
    /// ```
    pub fn of_str(int_number: &str) -> Option<Self> {
        std::str::FromStr::from_str(int_number).ok()
    }

    /// Return the digit at each position.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::int::Int;
    ///
    /// fn main() {
    ///     let i = Int::of(125436);
    ///     let digits = i.digits();
    ///     assert_eq!(digits, vec![1, 2, 5, 4, 3, 6]);
    /// }
    /// ```
    pub fn digits(&self) -> Vec<u8> {
        let string_view = self.inner.to_string();
        string_view
            .chars()
            .skip_while(|ch| !ch.is_ascii_digit())
            .map(|digit: char| digit as u8 - '0' as u8)
            .collect::<Vec<_>>()
    }

    /// Get the raw value of a int number.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::int::Int;
    ///
    /// fn main() {
    ///     let i = Int::of(69);
    ///     assert_eq!(i, 69);
    /// }
    /// ```
    pub fn raw(&self) -> i32 { self.inner }
}

/// Implement the concept of ZERO for the int number.
impl Zero for Int {
    fn zero() -> Self { Self { inner: 0i32 } }

    fn is_zero(&self) -> bool { self.inner == 0i32 }
}

/// Implement the concept of ONE for the int number.
impl One for Int {
    fn one() -> Self { Self { inner: 1i32 } }

    fn is_one(&self) -> bool { self.inner == 1i32 }
}

/// Implement Default for the int number.
impl std::default::Default for Int {
    fn default() -> Self { Self::zero() }
}

/// Implement the negation operation for the int number.
impl std::ops::Neg for Int {
    type Output = Self;

    fn neg(self) -> Self::Output { Self { inner: -self.inner } }
}

/// Implement the addition operation for the int number.
impl std::ops::Add for Int {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner + rhs.inner,
        }
    }
}

/// Implement the subtraction operation for the int number.
impl std::ops::Sub for Int {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner - rhs.inner,
        }
    }
}

/// Implement the multiplication operation for the int number.
impl std::ops::Mul for Int {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inner: self.inner * rhs.inner,
        }
    }
}

/// Implement the concept of NUMBER for the int number.
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

    /// Construct a int number from an integer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{int::Int, integer::Integer},
    ///     traits::number::Number,
    /// };
    ///
    /// fn main() {
    ///     let digits = (1..10).collect::<Vec<u8>>();
    ///     let integer = Integer::of(true, &digits);
    ///     let int = integer.map(|i| Int::from_integer(i));
    ///     assert_eq!(int, Some(Int::of(123456789)));
    /// }
    /// ```
    ///
    /// ## Panics
    ///
    /// If the size of the integer exceeds the maximum literal value of i32,
    /// it will cause a panic.
    ///
    /// ```rust,should_panic
    /// use rmatrix_ks::number::{
    ///     instances::{int::Int, integer::Integer},
    ///     traits::number::Number,
    /// };
    ///
    /// fn main() {
    ///     let integer = Integer::of_str("12345678910111213141516");
    ///     // Panic occurs here.
    ///     let _ = integer.map(|i| Int::from_integer(i));
    /// }
    /// ```
    fn from_integer(integer_number: Integer) -> Self {
        if integer_number.is_zero() {
            Self::zero()
        } else {
            let inner = format!("{:?}", integer_number)
                .parse::<i32>()
                .expect(&format!(
                    "Error[Int::from_Integer]: ({}) should be a valid i32 number.",
                    integer_number
                ));
            Self { inner }
        }
    }
}

/// Implement the concept of Real for Int.
impl Real for Int {
    fn to_rational(self) -> Rational
    where
        Self: crate::number::traits::integral::Integral,
    {
        Rational::of(self.to_integer(), Integer::one())
    }
}

/// Implement the concept of Integral for Int.
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
        Integer::of_str(&format!("{}", self)).expect(&format!(
            "Error[Int::to_integer]: ({}) should be a valid Integer.",
            self
        ))
    }
}

/// Implement Display for Int.
impl std::fmt::Display for Int {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

/// Implement Debug for Int.
impl std::fmt::Debug for Int {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+}", self.inner)
    }
}

/// Implement FromStr for Int.
impl std::str::FromStr for Int {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed_s = s.trim();
        if let Ok(num) = trimmed_s.parse::<i32>() {
            Ok(Self { inner: num })
        } else {
            eprintln!(
                "Error[Int::from_str]: ({}) is not a valid Int literal.",
                trimmed_s
            );
            Err(())
        }
    }
}

/// Uniform distribution of int numbers.
pub struct UniformI32(UniformInt<i32>);

/// Implement uniform sampling for the uniform distribution of int numbers.
impl UniformSampler for UniformI32 {
    type X = Int;

    fn new<B1, B2>(low: B1, high: B2) -> Result<UniformI32, rand::distr::uniform::Error>
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Ok(Self(UniformInt::<i32>::new(
            low.borrow().inner,
            high.borrow().inner,
        )?))
    }

    fn new_inclusive<B1, B2>(
        low: B1,
        high: B2,
    ) -> Result<UniformI32, rand::distr::uniform::Error>
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        Ok(Self(UniformInt::<i32>::new_inclusive(
            low.borrow().inner,
            high.borrow().inner,
        )?))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Self::X {
        Self::X::of(self.0.sample(rng))
    }
}

/// Implement uniform sampling for int numbers.
impl SampleUniform for Int {
    type Sampler = UniformI32;
}
