//! # number::utils
//!
//! Some util functions.

use crate::number::{
    instances::int::Int,
    traits::{fractional::Fractional, integral::Integral, number::Number, real::Real},
};

/// Division and modulus for i8
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::number::utils::i8_div_mod;
///
/// fn main() {
///     let a = -17i8;
///     let b = 10i8;
///     assert_eq!(i8_div_mod(a, b), (-2i8, 3i8))
/// }
/// ```
pub fn i8_div_mod(lhs: i8, rhs: i8) -> (i8, i8) {
    let quot = lhs / rhs;
    let rem = lhs % rhs;
    if rem == 0i8 || (quot >= 0i8 && rem > 0i8) {
        (quot, rem)
    } else {
        (quot - 1, lhs - (quot - 1) * rhs)
    }
}

/// Calculate the non-negative power of a number.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::number::instances::int::Int;
/// use rmatrix_ks::number::utils::non_negative_integral_power;
///
/// fn main() {
///     let a = Int::of(8i32);
///     let b = Int::of(3i32);
///
///     assert_eq!(non_negative_integral_power(a, b), Some(Int::of(512i32)));
/// }
/// ```
pub fn non_negative_integral_power<N: Number, I: Integral>(base: N, exponents: I) -> Option<N> {
    /// Optimize calculations using divide and conquer method.
    fn inner_power<N: Number, I: Integral>(base: N, exponents: I) -> N {
        if exponents.is_even() {
            inner_power(base.clone() * base, exponents.quotient(I::one() + I::one()))
        } else if exponents.is_one() {
            base
        } else {
            inner_power_acc(
                base.clone() * base.clone(),
                exponents.quotient(I::one() + I::one()),
                base,
            )
        }
    }

    /// Divide and conquer operation when the exponent is odd.
    fn inner_power_acc<N: Number, I: Integral>(base: N, exponents: I, acc: N) -> N {
        if exponents.is_even() {
            inner_power_acc(
                base.clone() * base,
                exponents.quotient(I::one() + I::one()),
                acc,
            )
        } else if exponents.is_one() {
            acc * base
        } else {
            inner_power_acc(
                base.clone() * base.clone(),
                exponents.quotient(I::one() + I::one()),
                acc * base,
            )
        }
    }

    // Main computation process.
    if exponents < I::zero() {
        // Exponentiation of integers does not support negative exponents.
        eprintln!(
            "Error[number::utils::non_negative_integral_power]: Negative exponents ({}) are not allowed.",
            exponents
        );
        None
    } else if exponents == I::zero() {
        // a^0 === 1.
        Some(N::one())
    } else if base == N::zero() {
        // 0^x === 0 where x != 0.
        Some(N::zero())
    } else {
        // Calculate using divide and conquer.
        Some(inner_power(base, exponents))
    }
}

/// Prevent overflow or underflow when encoding floating-point numbers.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::number::{instances::int::Int, utils::clamp};
///
/// fn main() {
///     let m = Int::of(-10);
///     let n = Int::of(5);
///     assert_eq!(clamp(m, n), Int::of(10));
/// }
/// ```
pub fn clamp(first: Int, second: Int) -> Int {
    (-first.clone()).max(first.min(second))
}

/// Calculate the integral power of a number.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::number::{
///     instances::{float::Float, int::Int},
///     traits::zero::Zero,
///     utils::integral_power,
/// };
///
/// fn main() {
///     let m = Int::of(-2);
///     let n = Float::of(2.0);
///     assert!((integral_power(n, m).is_some_and(|e| (e - Float::of(0.25)).is_zero())));
/// }
/// ```
pub fn integral_power<F: Fractional, I: Integral>(base: F, exponents: I) -> Option<F> {
    if exponents < I::zero() {
        non_negative_integral_power(base, -exponents).map(|v| v.reciprocal())
    } else {
        non_negative_integral_power(base, exponents)
    }
}

/// Calculate the greatest common divisor.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::number::{instances::int::Int, utils::gcd};
///
/// fn main() {
///     let m = Int::of(128);
///     let n = Int::of(96);
///     assert_eq!(gcd(m, n), Int::of(32));
/// }
/// ```
pub fn gcd<I: Integral>(lhs: I, rhs: I) -> I {
    fn inner_gcd<I: Integral>(lhs: I, rhs: I) -> I {
        if rhs == I::zero() || lhs == rhs {
            lhs
        } else {
            inner_gcd(rhs.clone(), lhs.remainder(rhs))
        }
    }
    inner_gcd(lhs.absolute_value(), rhs.absolute_value())
}

/// Calculate the least common multiple.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::number::{instances::int::Int, utils::lcm};
///
/// fn main() {
///     let m = Int::of(15);
///     let n = Int::of(6);
///     assert_eq!(lcm(m, n), Int::of(30));
/// }
/// ```
pub fn lcm<I: Integral>(lhs: I, rhs: I) -> I {
    if lhs.is_zero() || rhs.is_zero() {
        I::zero()
    } else {
        lhs.clone().quotient(gcd(lhs, rhs.clone())) * rhs
    }
}

/// Convert Integral to other number types.
///
/// ```rust
/// use rmatrix_ks::number::{
///     instances::{float::Float, int::Int},
///     utils::from_integral,
/// };
///
/// fn main() {
///     let m = Int::of(15);
///     let f = Float::of(15.0);
///     assert_eq!(from_integral::<Float, Int>(m), f);
/// }
/// ```
pub fn from_integral<N: Number, I: Integral>(integral_number: I) -> N {
    N::from_integer(integral_number.to_integer())
}

/// Convert Real to Fractional.
///
/// ```rust
/// use rmatrix_ks::number::{
///     instances::{double::Double, ratio::Rational},
///     utils::real_to_frac,
/// };
///
/// fn main() {
///     let m = Rational::of_str("12 % 5").unwrap();
///     let f = Double::of(2.4);
///     assert_eq!(real_to_frac::<Rational, Double>(m), f);
/// }
/// ```
pub fn real_to_frac<R: Real, F: Fractional>(real_number: R) -> F {
    F::from_rational(real_number.to_rational())
}
