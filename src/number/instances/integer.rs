//! # instances::integer
//!
//! Functions and implementations related to signed integers.

use crate::number::{
    instances::ratio::Rational,
    traits::{integral::Integral, number::Number, one::One, real::Real, zero::Zero},
    utils::i8_div_mod,
};

/// Integer type with no size limit.
#[derive(Clone, PartialEq, Eq)]
pub struct Integer {
    /// Sign of the number.
    pub sign: bool,
    inner: Vec<u8>,
}

impl Integer {
    /// Construct an integer by passing the sign and each digit.
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
    ///
    /// ## Warning
    ///
    /// <div class="warning">
    ///
    /// Each digit should be between [0, 9],
    /// which means it is a single decimal digit.
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::integer::Integer;
    ///
    /// fn main() {
    ///     let digits = vec![1, 2, 10];
    ///     assert_eq!(Integer::of(true, &digits), None);
    /// }
    /// ```
    ///
    /// </div>
    pub fn of(sign: bool, digits: &[u8]) -> Option<Self> {
        // Remove leading zeros.
        let mut inner = digits
            .iter()
            .skip_while(|&&digit| digit == 0u8)
            .map(|&digit| digit)
            .collect::<Vec<u8>>();
        if inner.is_empty() {
            Some(Self::zero())
        } else {
            // Store in reverse for easier computation.
            inner.reverse();
            // Each digit should be between [0, 9].
            if inner.iter().all(|&digit| digit < 10u8) {
                Some(Self { sign, inner })
            } else {
                eprintln!(
                    concat!(
                        "Error[Integer::of]: ",
                        "Each digit of the integer ",
                        "should be within the range [0, 9] ({:?})."
                    ),
                    digits,
                );
                None
            }
        }
    }

    /// Construct integer numbers from string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::integer::Integer;
    ///
    /// fn main() {
    ///     let i1 = Integer::of_str("-123456");
    ///     let i2 = Integer::of(false, &[1, 2, 3, 4, 5, 6]);
    ///     assert_eq!(i1, i2);
    /// }
    /// ```
    pub fn of_str(integer_number: &str) -> Option<Self> {
        std::str::FromStr::from_str(integer_number).ok()
    }

    /// Return the digit at each position.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::integer::Integer;
    ///
    /// fn main() {
    ///     let digits = Integer::of_str("-123456").map(|e| e.digits());
    ///     assert_eq!(digits, Some(vec![1, 2, 3, 4, 5, 6]));
    /// }
    /// ```
    pub fn digits(&self) -> Vec<u8> { self.inner.iter().rev().cloned().collect::<Vec<u8>>() }

    /// Non-negative integer addition.
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
                "Error[Integer::integer_add]: Each digit should be within the range [1, 9].",
            )
        }
    }

    /// Non-negative integer subtraction.
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
                .rev() // Revert to normal order.
                .map(|&digit| digit as u8)
                .collect::<Vec<u8>>();
            Self::of(sign, &digits).expect(
                "Error[Integer::integer_sub]: Each digit should be within the range [1, 9].",
            )
        }
    }

    /// Non-negative integer comparison.
    fn integer_cmp(&self, rhs: &Integer) -> std::cmp::Ordering {
        if self.inner.len() > rhs.inner.len() {
            std::cmp::Ordering::Greater
        } else if self.inner.len() < rhs.inner.len() {
            std::cmp::Ordering::Less
        } else {
            for (lhs_digit, rhs_digit) in self.digits().iter().zip(rhs.digits().iter()) {
                if lhs_digit > rhs_digit {
                    return std::cmp::Ordering::Greater;
                } else if lhs_digit < rhs_digit {
                    return std::cmp::Ordering::Less;
                }
            }
            std::cmp::Ordering::Equal
        }
    }
}

/// Implement the concept of ZERO for the integer number.
impl Zero for Integer {
    /// Retrieve zeros in integers.
    ///
    /// # Examples
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

    /// Validate whether an integer is zero.
    ///
    /// # Examples
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
    fn is_zero(&self) -> bool { self.inner.is_empty() || self.inner.iter().all(|&e| e == 0u8) }
}

impl One for Integer {
    fn one() -> Self {
        Self {
            sign: true,
            inner: vec![1u8],
        }
    }

    /// Validate whether an integer is one.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::{instances::integer::Integer, traits::one::One};
    ///
    /// fn main() {
    ///     let a = Integer::one();
    ///     assert!(a.is_one());
    ///     let b = Integer::of(true, &[1, 6]);
    ///     assert!(b.is_some_and(|e| !e.is_one()));
    /// }
    /// ```
    fn is_one(&self) -> bool {
        (!self.inner.is_empty()) // An empty integer cannot be one.
            && (self
                .inner
                .iter()
                .rev() // Revert to normal order.
                .skip_while(|&&e| e == 0)
                .collect::<Vec<&u8>>()
                == vec![&1u8]) // Exclude the influence of leading zeros.
    }
}

/// Implement Default for the integer number.
impl std::default::Default for Integer {
    fn default() -> Self { Self::zero() }
}

/// Implement PartialOrd for the integer number.
impl std::cmp::PartialOrd for Integer {
    /// Compare two integers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::number::instances::integer::Integer;
    ///
    /// fn main() {
    ///     // 512
    ///     let a = Integer::of(true, &[5, 1, 2]).unwrap();
    ///     // 512
    ///     let another_a = Integer::of(true, &[5, 1, 2]).unwrap();
    ///     // -128
    ///     let b = Integer::of(false, &[1, 2, 8]).unwrap();
    ///     // 1024
    ///     let c = Integer::of(true, &[1, 0, 2, 4]).unwrap();
    ///     assert_eq!(a.partial_cmp(&another_a), Some(std::cmp::Ordering::Equal));
    ///     assert_eq!(a.partial_cmp(&b), Some(std::cmp::Ordering::Greater));
    ///     assert_eq!(a.partial_cmp(&c), Some(std::cmp::Ordering::Less));
    /// }
    /// ```
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.is_zero() && other.is_zero() {
            Some(std::cmp::Ordering::Equal)
        } else {
            match (self.sign, other.sign) {
                (true, false) => Some(std::cmp::Ordering::Greater),
                (false, true) => Some(std::cmp::Ordering::Less),
                (true, true) => Some(self.integer_cmp(other)),
                (false, false) => Some(other.integer_cmp(self)),
            }
        }
    }
}

/// Implement Ord for the integer number.
impl std::cmp::Ord for Integer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other)
            .expect("Error[Integer::cmp]: Integer should be ordered.")
    }
}

/// Implement the negation operation for the integer number.
impl std::ops::Neg for Integer {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            sign: if self.is_zero() { true } else { !self.sign },
            inner: self.inner,
        }
    }
}

/// Implement the addition operation for the integer number.
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

/// Implement the subtraction operation for the integer number.
impl std::ops::Sub for Integer {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output { self + (-rhs) }
}

/// Implement the multiplication operation for the integer number.
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
            Self::of(self.sign == rhs.sign, &product)
                .expect("Error[Integer::mul]: Each digit should be within the range [1, 9].")
        }
    }
}

/// Implement the concept of NUMBER for the integer number.
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

    fn from_integer(integer_number: Integer) -> Self { integer_number }
}

/// Implement the concept of Real for Integer.
impl Real for Integer {
    fn to_rational(self) -> Rational { Rational::of(self, Self::one()) }
}

/// Implement the concept of Integral for Integer.
impl Integral for Integer {
    fn quot_rem(self, rhs: Self) -> (Self, Self) {
        if self == rhs {
            // x / x => 1, include x = 0
            (Self::one(), Self::zero())
        } else if rhs.is_zero() {
            // x / 0 => error
            panic!("Error[Integer::quot_rem]: divide by zero");
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
                    "Error[Integer::quot_rem]: ({}, {}) is out of bounds.",
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
                .expect(concat!(
                    "Error[Integer::quot_rem]: ",
                    "Each digit should be within the range [1, 9]."
                ))
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

    fn to_integer(self) -> Integer { self }
}

/// Implement Display for Integer.
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

/// Implement Debug for Integer.
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

/// Implement FromStr for Integer.
impl std::str::FromStr for Integer {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Remove any leading and trailing whitespace characters from the string.
        let trimmed_s = s.trim();
        // Use regular expressions to validate the string.
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
            .ok_or_else(|| {
                eprintln!(
                    "Error[Ineteger::from_str]: Failed to parse {} from the string.",
                    trimmed_s
                )
            })
    }
}
