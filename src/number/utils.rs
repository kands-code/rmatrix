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

/// Generate all permutations of the given list.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::number::utils::permutation;
///
/// fn main() {
///     let mut p = permutation(&[1, 2, 3]);
///     p.sort();
///     let p_expect = vec![
///         vec![1, 2, 3],
///         vec![1, 3, 2],
///         vec![2, 1, 3],
///         vec![2, 3, 1],
///         vec![3, 1, 2],
///         vec![3, 2, 1],
///     ];
///     assert_eq!(p, p_expect);
/// }
/// ```
pub fn permutation<T>(elements: &[T]) -> Vec<Vec<T>>
where
    T: Clone,
{
    let mut exts = Vec::new();
    if !elements.is_empty() {
        for idx in 0..elements.len() {
            let remain = [elements[..idx].to_vec(), elements[(idx + 1)..].to_vec()].concat();
            let ps = permutation(&remain);
            if ps.is_empty() {
                exts.push(vec![elements[idx].clone()]);
            } else {
                for p in permutation(&remain) {
                    let mut ext = p.clone();
                    ext.push(elements[idx].clone());
                    exts.push(ext);
                }
            }
        }
    }
    exts
}

/// Calculate the number of inversions in the given list.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::number::utils::inversion_count;
///
/// fn main() {
///     let p1 = [1, 3, 2];
///     assert_eq!(inversion_count(&p1), 1);
///     let p2 = [1, 2, 3];
///     assert_eq!(inversion_count(&p2), 0);
///     assert_eq!(inversion_count::<usize>(&[]), 0);
/// }
/// ```
pub fn inversion_count<T>(elements: &[T]) -> usize
where
    T: Clone + PartialOrd,
{
    let mut exchange = 0;
    if !elements.is_empty() {
        let mut cloned = elements.to_vec();
        for idx in 0..(elements.len() - 1) {
            let mut min = idx;
            for p in (idx + 1)..elements.len() {
                if cloned[p] < cloned[min] {
                    min = p;
                }
            }
            if min != idx {
                (cloned[idx], cloned[min]) = (cloned[min].clone(), cloned[idx].clone());
                exchange = exchange + 1;
            }
        }
    }
    exchange
}

/// Calculate the non-negative power of a number.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::number::{instances::int::Int, utils::non_negative_integral_power};
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
pub fn clamp(first: Int, second: Int) -> Int { (-first.clone()).max(first.min(second)) }

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
/// Represented in two's complement form.
///
/// The first element of the returned list is the sign bit,
/// and the remaining elements represent the binary expression of its absolute value.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::number::{instances::int::Int, utils::integral_to_binary};
///
/// fn main() {
///     let i1 = Int::of(123);
///     let i1_digits = integral_to_binary(i1);
///     //  123 = 0b0111011
///     assert_eq!(i1_digits, vec![0, 1, 1, 1, 1, 0, 1, 1]);
///     let i2 = Int::of(-123456);
///     // -123456 = 0b100001110111000000
///     let i2_digits = integral_to_binary(i2);
///     assert_eq!(
///         i2_digits,
///         vec![1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
///     );
/// }
/// ```
pub fn integral_to_binary<I: Integral>(int_val: I) -> Vec<u8> {
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
                            "Error[number::utils::decimal_to_binary]: ",
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

    let mut bin = Vec::new();
    let val = int_val.absolute_value();
    integral_to_binary_inner(val, &mut bin);
    bin.insert(0, 0u8);
    if int_val < I::zero() {
        let mut complement = vec![0u8; bin.len()];
        let mut sup = 1u8;
        for (idx, e) in bin
            .iter()
            .map(|&e| if e == 1u8 { 0u8 } else { 1u8 })
            .enumerate()
            .rev()
        {
            if sup == 1u8 && e == 1u8 {
                complement[idx] = 0u8;
            } else if sup == 1u8 && e == 0u8 {
                complement[idx] = 1u8;
                sup = 0u8;
            } else if sup == 1u8 && idx == 0usize {
                // Prevent overflow from overwriting the sign bit.
                complement.insert(1usize, 1u8);
            } else {
                complement[idx] = e;
            }
        }
        bin = complement;
    }
    bin
}
