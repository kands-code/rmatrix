//! # Number Utils
//!
//! some useful tools

use crate::number::{
    instances::int::Int,
    traits::{fractional::Fractional, integral::Integral, number::Number, real::Real},
};

/// division and modulus for i8
///
/// # Example
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

/// raise a number to a non-negative integral power
///
/// # Example
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
    /// use divide and conquer method to optimize the calculation process
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

    /// divide and conquer for the case where the exponent is an odd number
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

    // main process
    if exponents < I::zero() {
        // integeral exponents cannot handle negative exponents
        eprintln!(
            "Error[utils::non_negative_integral_power]: negative exponents {}",
            exponents
        );
        None
    } else if exponents == I::zero() {
        // a^0 === 1
        Some(N::one())
    } else if base == N::zero() {
        // 0^x === 0 where x != 0
        Some(N::zero())
    } else {
        // use a divide and conquer method
        Some(inner_power(base, exponents))
    }
}

/// prevent exponent over/underflow when encoding floating point numbers
pub fn clamp(first: Int, second: Int) -> Int {
    let smaller = if first < second {
        first.clone()
    } else {
        second
    };
    let negtive_first = first;
    if negtive_first > smaller {
        negtive_first
    } else {
        smaller
    }
}

/// raise a number to an integral power
pub fn integeral_power<F: Fractional, I: Integral>(base: F, exponents: I) -> Option<F> {
    if exponents < I::zero() {
        non_negative_integral_power(base, -exponents).map(|v| v.reciprocal())
    } else {
        non_negative_integral_power(base, exponents)
    }
}

/// the greatest common factor
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

/// the least common multiple
pub fn lcm<I: Integral>(lhs: I, rhs: I) -> I {
    if lhs.is_zero() || rhs.is_zero() {
        I::zero()
    } else {
        lhs.clone().quotient(gcd(lhs, rhs.clone())) * rhs
    }
}

/// convert integeral number to other number type
pub fn from_integeral<N: Number, I: Integral>(integeral_number: I) -> N {
    N::from_integer(integeral_number.to_integer())
}

/// convert real number to fractional number
pub fn real_to_frac<R: Real, F: Fractional>(real_number: R) -> F {
    F::from_rational(real_number.to_rational())
}
