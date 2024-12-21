//! # number::utils
//!
//! Some util functions.

use crate::number::{
    instances::{int::Int, integer::Integer},
    traits::{
        fractional::Fractional, integral::Integral, number::Number, one::One, real::Real,
        realfloat::RealFloat,
    },
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
            concat!(
                "Error[number::utils::non_negative_integral_power]: ",
                "Negative exponents ({}) are not allowed."
            ),
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

/// Convert an integer to binary format.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::number::{instances::int::Int, utils::integral_to_binary};
///
/// fn main() {
///     let i = Int::of(123);
///     let i_digits = integral_to_binary(i);
///     assert_eq!(i_digits, Some(vec![1, 1, 1, 1, 0, 1, 1]));
/// }
/// ```
pub fn integral_to_binary<I: Integral>(int_val: I) -> Option<Vec<u8>> {
    fn integral_to_binary_inner<I: Integral>(int: I, acc: &mut Vec<u8>) {
        if int == I::zero() {
            acc.insert(0, 0u8);
        } else if int == I::one() {
            acc.insert(0, 1u8);
        } else {
            let two: I = I::one() + I::one();
            let (d, m) = int.div_mod(two);
            format!("{}", m).chars().nth(0).map_or_else(
                || {
                    eprintln!(
                        concat!(
                            "Error[number::utils::decimal_to_binary]:",
                            "Failed to retrieve the value of the modulus ({})"
                        ),
                        m
                    )
                },
                |c| acc.insert(0, c as u8 - '0' as u8),
            );
            integral_to_binary_inner(d, acc);
        }
    }

    if int_val < I::zero() {
        eprintln!(concat!(
            "Error[number::utils::decimal_to_binary]:",
            "Cannot convert a negative number to binary format"
        ));
        None
    } else {
        let mut bin = Vec::new();
        integral_to_binary_inner(int_val, &mut bin);
        Some(bin)
    }
}

/// Convert a floating-point number to binary format.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::number::{instances::double::Double, utils::decimal_to_binary};
///
/// fn main() {
///     let d = Double::of(3.14);
///     assert_eq!(
///         decimal_to_binary(d),
///         Some((
///             vec![1, 1],
///             vec![
///                 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1,
///                 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1
///             ]
///         ))
///     );
/// }
/// ```
pub fn decimal_to_binary<F: RealFloat>(rfp_val: F) -> Option<(Vec<u8>, Vec<u8>)> {
    /// Internal implementation of converting floating-point number to binary format.
    ///
    /// # Input
    ///
    /// - Real floating-point number to convert.
    /// - Remaining available bits.
    /// - Number of bits already inserted.
    /// - Accumulative result cache
    fn decimal_to_binary_inner<F: RealFloat>(rfp: F, rem: usize, ins: usize, acc: &mut Vec<u8>) {
        if rfp.is_zero() {
            acc.resize_with(rem, || 0u8);
        } else {
            let two = F::one() + F::one();
            let (integral_part, fractional_part) = (rfp * two).proper_fraction::<Integer>();
            if ins > rem && integral_part.is_one() {
                let mut stop = false;
                for index in (0..acc.len()).rev() {
                    if stop {
                        break;
                    } else if acc[index] == 1u8 {
                        acc[index] = 0u8;
                    } else {
                        acc[index] = 1u8;
                        stop = true;
                    }
                }
            } else {
                acc.push(if integral_part.is_one() { 1u8 } else { 0u8 });
                decimal_to_binary_inner(fractional_part, rem, ins + 1, acc);
            }
        }
    }

    let (integral_part, fractional_part) = rfp_val.proper_fraction::<Integer>();
    let integral_digits = integral_to_binary(integral_part).expect(
        "Error[number::utils::decimal_to_binary]: Failed to convert integer to binary format.",
    );
    let total_digits = format!("{:?}", F::FLOAT_DIGITS)
        .parse::<usize>()
        .expect("Error[number::utils::decimal_to_binary]: Failed to convert FLOAT_DIGITS to usize");
    if integral_digits.len() > total_digits {
        eprintln!(
            concat!(
                "Error[number::utils::decimal_to_binary]: ",
                "The number of digits in the integer part ({}) ",
                "exceeds FLOAT_DIGITS limits ({}), conversion not possible"
            ),
            integral_digits.len(),
            total_digits,
        );
        None
    } else {
        let mut float_part = Vec::new();
        decimal_to_binary_inner(
            fractional_part,
            total_digits - integral_digits.len(),
            0,
            &mut float_part,
        );
        Some((integral_digits, float_part))
    }
}
