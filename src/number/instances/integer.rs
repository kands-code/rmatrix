//! # Number Type :: Integer
//!
//! Integer of arbitrary length.

use crate::number::instances::ratio::Rational;
use crate::number::traits::integeral::Integral;
use crate::number::traits::number::Number;
use crate::number::traits::one::One;
use crate::number::traits::real::Real;
use crate::number::traits::zero::Zero;
use crate::number::utils::i8_div_mod;

/// Integer
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "serde_mat", derive(serde::Deserialize, serde::Serialize))]
pub struct Integer {
    pub sign: bool,
    inner: Vec<u8>,
}

impl Integer {
    /// default create function for Integer
    ///
    /// # Note
    ///
    /// every digits should be in [0, 10)
    ///
    /// # Example
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::integer::Integer;
    ///
    /// fn main() {
    ///     let digits = vec![1, 2, 3];
    ///     // Integer(123)
    ///     let Some(_) = Integer::of(true, &digits) else {
    ///         unreachable!();
    ///     };
    /// }
    /// ```
    pub fn of(sign: bool, digits: &[u8]) -> Option<Self> {
        // remove redundant zeros
        let mut inner = digits
            .iter()
            .skip_while(|&&digit| digit == 0u8)
            .map(|&digit| digit)
            .collect::<Vec<u8>>();
        if inner.is_empty() {
            Some(Self::zero())
        } else {
            // reverse storage for ease of calculation
            inner.reverse();
            // every digits should be in [0, 10)
            if inner.iter().all(|&digit| digit < 10u8) {
                Some(Self { sign, inner })
            } else {
                eprintln!(
                    "Error[rmatrix_ks::number::instances::integer::Integer::of]: \
                    every digits should be in [0, 10) {:?}",
                    digits,
                );
                None
            }
        }
    }

    /// create Integer from string literal
    ///
    /// # Note
    ///
    /// the string literal should match `r"([+-]?)([0-9_]+)"`
    ///
    /// # Example
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::integer::Integer;
    ///
    /// fn main() {
    ///     let integer_str = "-123_4567";
    ///     // Integer(-1234567)
    ///     let Some(_) = Integer::of_str(integer_str) else {
    ///         unreachable!();
    ///     };
    /// }
    /// ```
    pub fn of_str(integer_number: &str) -> Option<Self> {
        std::str::FromStr::from_str(integer_number).ok()
    }

    /// get the digits of the integer
    ///
    /// # Example
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::integer::Integer;
    ///
    /// fn main() {
    ///     if let Some(integer_number) = Integer::of_str("-123_4567") {
    ///         assert_eq!(integer_number.digits(), vec![1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8]);
    ///     };
    /// }
    /// ```
    pub fn digits(&self) -> Vec<u8> {
        self.inner.iter().rev().cloned().collect::<Vec<u8>>()
    }

    /// add self and rhs with non-negative
    fn integer_add(self, rhs: Self) -> Self {
        if self.is_zero() {
            rhs
        } else if rhs.is_zero() {
            self
        } else {
            let expect_capacity = self.inner.len().max(rhs.inner.len()) + 1usize;
            let mut sum: Vec<u8> = vec![0u8; expect_capacity];
            let mut carry = 0u8;
            for idx in 0..expect_capacity {
                let factor = self.inner.get(idx).map_or(0u8, |&digit| digit)
                    + rhs.inner.get(idx).map_or(0u8, |&digit| digit)
                    + carry;
                sum[idx] = factor % 10;
                carry = factor / 10;
            }
            sum.reverse();
            Self::of(true, &sum).expect(
                "Error[rmatrix_ks::number::instances::integer::Integer::integer_add]: \
                every digits should be in [0, 10)",
            )
        }
    }

    /// subtract self and rhs with non-negative
    fn integer_sub(self, rhs: Self) -> Self {
        if self.is_zero() {
            -rhs
        } else if rhs.is_zero() {
            self
        } else {
            let expect_capacity = self.inner.len().max(rhs.inner.len());
            let mut diff: Vec<i8> = vec![0i8; expect_capacity];
            let (mut sign, subtracted, subtracting) = if self > rhs {
                (true, self, rhs)
            } else {
                (false, rhs, self)
            };
            let mut carry = 0i8;
            for idx in 0..expect_capacity {
                let factor = subtracted.inner.get(idx).map_or(0i8, |&digit| digit as i8)
                    - subtracting.inner.get(idx).map_or(0i8, |&digit| digit as i8)
                    + carry;
                (carry, diff[idx]) = i8_div_mod(factor, 10i8);
            }
            sign = (carry == 0i8) == sign;
            let digits = diff
                .iter()
                .rev() // convert to noraml digits
                .map(|&digit| digit as u8)
                // .map(|&digit| digit as u8)
                .collect::<Vec<u8>>();
            Self::of(sign, &digits).expect(
                "Error[rmatrix_ks::number::instances::integer::Integer::integer_sub]: \
                every digits should be in [0, 10)",
            )
        }
    }
}

impl Zero for Integer {
    /// ZERO for Integer
    ///
    /// # Example
    ///
    /// ```rust
    /// use rmatrix_ks::number::{instances::integer::Integer, traits::zero::Zero};
    ///
    /// fn main() {
    ///     let a = Integer::zero();
    ///     let Some(zero) = Integer::of_str("0") else {
    ///         unreachable!();
    ///     };
    ///     assert_eq!(a, zero);
    /// }
    /// ```
    fn zero() -> Self {
        Self {
            sign: true,
            inner: vec![0u8],
        }
    }

    /// checks if it is the ZERO
    ///
    /// # Example
    ///
    /// ```rust
    /// use rmatrix_ks::number::{instances::integer::Integer, traits::zero::Zero};
    ///
    /// fn main() {
    ///     let a = Integer::zero();
    ///     assert!(a.is_zero());
    ///     let Some(b) = Integer::of(true, &[1, 6]) else {
    ///         unreachable!();
    ///     };
    ///     assert!(!b.is_zero());
    /// }
    /// ```
    fn is_zero(&self) -> bool {
        self.inner.len() == 1 && self.inner.get(0).is_some_and(|&v| v == 0u8)
    }
}

impl One for Integer {
    /// ONE for Integer
    ///
    /// # Example
    ///
    /// ```rust
    /// use rmatrix_ks::number::{instances::integer::Integer, traits::one::One};
    ///
    /// fn main() {
    ///     let a = Integer::one();
    ///     let Some(one) = Integer::of_str("1") else {
    ///         unreachable!();
    ///     };
    ///     assert_eq!(a, one);
    /// }
    /// ```
    fn one() -> Self {
        Self {
            sign: true,
            inner: vec![1u8],
        }
    }

    /// checks if it is the ONE
    ///
    /// # Example
    ///
    /// ```rust
    /// use rmatrix_ks::number::{instances::integer::Integer, traits::one::One};
    ///
    /// fn main() {
    ///     let a = Integer::one();
    ///     assert!(a.is_one());
    ///     let Some(b) = Integer::of(true, &[1, 6]) else {
    ///         unreachable!();
    ///     };
    ///     assert!(!b.is_one());
    /// }
    /// ```
    fn is_one(&self) -> bool {
        self.inner.len() == 1 && self.inner.get(0).is_some_and(|&v| v == 1u8)
    }
}

impl std::cmp::PartialOrd for Integer {
    /// compare two integers
    ///
    /// # Example
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::integer::Integer;
    ///
    /// fn main() {
    ///     // 512
    ///     let a = Integer::of(true, &[5, 1, 2]);
    ///     // 512
    ///     let another_a = Integer::of(true, &[5, 1, 2]);
    ///     // -128
    ///     let b = Integer::of(false, &[1, 2, 8]);
    ///     // 1024
    ///     let c = Integer::of(true, &[1, 0, 2, 4]);
    ///     assert_eq!(a.partial_cmp(&another_a), Some(std::cmp::Ordering::Equal));
    ///     assert_eq!(a.partial_cmp(&b), Some(std::cmp::Ordering::Greater));
    ///     assert_eq!(a.partial_cmp(&c), Some(std::cmp::Ordering::Less));
    /// }
    /// ```
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.inner.len() > other.inner.len() {
            Some(std::cmp::Ordering::Greater)
        } else if self.inner.len() < other.inner.len() {
            Some(std::cmp::Ordering::Less)
        } else {
            for (self_digit, other_digit) in self.digits().iter().zip(other.digits().iter()) {
                if self_digit > other_digit {
                    return Some(std::cmp::Ordering::Greater);
                } else if self_digit < other_digit {
                    return Some(std::cmp::Ordering::Less);
                }
            }
            Some(std::cmp::Ordering::Equal)
        }
    }
}

impl std::ops::Neg for Integer {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            sign: if self.is_zero() { true } else { !self.sign },
            inner: self.inner,
        }
    }
}

impl std::ops::Add for Integer {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self.sign, rhs.sign) {
            (true, false) => self.integer_sub(rhs.absolute_value()),
            (false, true) => rhs.integer_sub(self.absolute_value()),
            (true, true) => self.integer_add(rhs),
            (false, false) => -self.absolute_value().integer_add(rhs.absolute_value()),
        }
    }
}

impl std::ops::Sub for Integer {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Mul for Integer {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            Self::zero()
        } else {
            let expect_capacity = self.inner.len() + rhs.inner.len();
            let mut product_layers = vec![vec![0u8; expect_capacity]; rhs.inner.len()];
            for (idx, &digit) in rhs.inner.iter().enumerate() {
                for (loc, &p) in self.inner.iter().enumerate() {
                    product_layers[idx][loc + idx] = digit * p;
                }
            }
            let mut product = vec![0u8; expect_capacity];
            let mut carry = 0u8;
            for idx in 0usize..expect_capacity {
                let factor = product_layers.iter().map(|layer| layer[idx]).sum::<u8>() + carry;
                product[idx] = factor % 10u8;
                carry = factor / 10u8;
            }
            product.reverse();
            Self::of(self.sign == rhs.sign, &product).expect(
                "Error[rmatrix_ks::number::instances::integer::Integer::mul]: \
                every digits should be in [0, 10)",
            )
        }
    }
}

impl Number for Integer {
    fn absolute_value(&self) -> Self {
        Self {
            sign: true,
            inner: self.inner.clone(),
        }
    }

    fn sign_number(&self) -> Self {
        if self.is_zero() {
            Self::zero()
        } else if self.sign {
            Self::one()
        } else {
            -Self::one()
        }
    }

    fn from_integer(integer_number: Integer) -> Self {
        integer_number
    }
}

impl Real for Integer {
    fn to_rational(self) -> Rational {
        Rational::of(self, Self::one())
    }
}

impl Integral for Integer {
    fn quot_rem(self, rhs: Self) -> (Self, Self) {
        if self == rhs {
            // x / x => 1, include x = 0
            (Self::one(), Self::zero())
        } else if rhs.is_zero() {
            // x / 0 => error
            panic!(
                "Error[rmatrix_ks::number::instances::integer::Integer::quot_rem]: \
                divide by zero"
            );
        } else if self.is_zero() {
            // 0 / x => 0 ... 0
            (Self::zero(), Self::zero())
        } else if rhs.absolute_value().is_one() {
            // x / 1 => x ... 0 and x / (-1) => -x ... 0
            (if rhs.is_one() { self } else { -self }, Self::zero())
        } else {
            // x / y => m ... n where m * y + n => x
            let self_digits = self.digits();
            let rhs_abs = rhs.absolute_value();
            let mut head = 0usize;
            let mut expand = 1usize;
            let mut quot_digits = Vec::<u8>::new();
            let mut rem_digits = Vec::<u8>::new();
            while head + expand <= self_digits.len() {
                let mut rem_buffer = rem_digits.clone();
                rem_buffer.extend(self_digits[head..(head + expand)].iter());
                let buffer = Self::of(true, &rem_buffer).expect(&format!(
                    "Error[rmatrix_ks::number::instances::integer::Integer::quot_rem]: \
                        out of boundary ({}, {})",
                    head,
                    head + expand,
                ));
                if buffer < rhs_abs {
                    quot_digits.push(0u8);
                    if head + expand == self_digits.len() {
                        rem_digits = buffer.digits();
                        break;
                    } else {
                        expand = expand + 1;
                    }
                } else {
                    let mut factor = Self::one();
                    while (factor.clone() + Self::one()) * rhs_abs.clone() <= buffer {
                        factor = factor + Self::one();
                    }
                    rem_digits = (buffer - factor.clone() * rhs_abs.clone()).digits();
                    quot_digits.push(factor.inner[0]);
                    head = head + expand;
                    expand = 1usize;
                }
            }
            Self::of(self.sign == rhs.sign, &quot_digits)
                .zip(Self::of(self.sign, &rem_digits))
                .expect(
                    "Error[rmatrix_ks::number::instances::integer::Integer::quot_rem]: \
                    every digits should be in [0, 10)",
                )
        }
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
        self
    }
}

impl std::fmt::Display for Integer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}",
            if self.sign { "" } else { "-" },
            self.inner
                .iter()
                .rev()
                .map(|d| (d + ('0' as u8)) as char)
                .collect::<String>()
        )
    }
}

impl std::fmt::Debug for Integer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}",
            if self.sign { "+" } else { "-" },
            self.inner
                .iter()
                .rev()
                .map(|d| (d + ('0' as u8)) as char)
                .collect::<String>()
        )
    }
}

impl std::str::FromStr for Integer {
    type Err = ();

    /// convert string literal to Integer
    ///
    /// note and example see [Integer::of_str()]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // remove extra whitespaces
        let trimmed_s = s.trim();
        // use regular expressions to validate string
        regex::Regex::new(r"(?<sign>[+-]?)(?<digits>[0-9_]+)")
            .ok()
            .and_then(|matcher| {
                let matched_captures = matcher.captures(trimmed_s);
                if let Some(captures) = matched_captures {
                    let sign_str = &captures["sign"];
                    let sign = sign_str.is_empty() || sign_str == "+";
                    let digits_str = &captures["digits"];
                    let digits = digits_str
                        .chars()
                        .filter(|digit| digit.is_ascii_digit())
                        .map(|digit| (digit as u8) - ('0' as u8))
                        .collect::<Vec<u8>>();
                    Self::of(sign, &digits)
                } else {
                    None
                }
            })
            .ok_or(())
    }
}
