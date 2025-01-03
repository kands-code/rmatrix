//! # traits::realfloat
//!
//! Types that implement this trait can be considered as real floating-point numbers.

use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::number::{
    instances::{int::Int, integer::Integer},
    traits::{floating::Floating, one::One, realfrac::RealFrac, zero::Zero},
    utils::{clamp, from_integral, integral_power, non_negative_integral_power},
};

/// Concepts of RealFloat.
pub trait RealFloat: RealFrac + Floating {
    /// The base of the numerical system.
    ///
    /// The base, also known as "radix" in standards,
    /// is typically `2`, which represents binary representation.
    /// However, for decimal numbers, the base may be `10`.
    const FLOAT_RADIX: Int = Int::of(2);

    /// Number of digits in the radix used, including any implicit digit, but not counting the sign bit.
    const FLOAT_DIGITS: Int;

    /// In the standard representation of floating-point numbers,
    /// the range of the exponent is defined as `[-m + 2, m + 1)`
    /// if the maximum value of the floating-point exponent is `m`.
    const FLOAT_RANGE: (Int, Int);

    /// Decode a real floating-point number into its significand and exponent.
    ///
    /// # Examples
    ///
    /// For example, for the `Double`:
    ///
    /// ```rust
    /// use rmatrix_ks::number::{
    ///     instances::{double::Double, int::Int, integer::Integer},
    ///     traits::realfloat::RealFloat,
    /// };
    ///
    /// fn main() {
    ///     let d1 = Double::of(3.14);
    ///     assert_eq!(
    ///         d1.decode_float(),
    ///         (Integer::of_str("7070651414971679").unwrap(), Int::of(-51))
    ///     );
    ///
    ///     let d2 = Double::of(-13.14);
    ///     assert_eq!(
    ///         d2.decode_float(),
    ///         (Integer::of_str("-7397162387956040").unwrap(), Int::of(-49))
    ///     );
    /// }
    /// ```
    fn decode_float(self) -> (Integer, Int) {
        let sign = self >= Self::zero();
        let mut rfp = self.absolute_value();
        let mut exponent = Int::of(0);
        let rfp_two = Self::one() + Self::one();
        // Convert the floating-point number to
        // a base-2 exponential product representation,
        // i.e., m * 2^p.
        while rfp > Self::one() {
            let next_rfp = rfp.clone() * Self::half();
            if next_rfp < Self::one() {
                break;
            } else {
                rfp = next_rfp;
                exponent = exponent + Int::one();
            }
        }
        while rfp < Self::one() {
            rfp = rfp.clone() * rfp_two.clone();
            exponent = exponent - Int::one();
        }
        exponent = exponent - Self::FLOAT_DIGITS + Int::one();
        let mut float_digits = Vec::new();
        rfp = rfp - Self::one();
        let mut count_digits = Int::zero();
        while count_digits < Self::FLOAT_DIGITS - Int::one() {
            rfp = rfp * rfp_two.clone();
            if rfp >= Self::one() {
                float_digits.push(1u8);
                rfp = rfp - Self::one();
            } else {
                float_digits.push(0u8);
            }
            count_digits = count_digits + Int::one();
        }
        let integer_two = Integer::one() + Integer::one();
        let mut significand = float_digits
            .iter()
            .rev()
            .enumerate()
            .par_bridge()
            .map(|(idx, &e)| {
                if e == 1u8 {
                    non_negative_integral_power(integer_two.clone(), Int::of(idx as i32))
                        .expect(&format!(
                            concat!(
                                "Error[RealFloat::decode_float]: ",
                                "Failed to compute pow(2, {})"
                            ),
                            idx
                        ))
                } else {
                    Integer::zero()
                }
            })
            .reduce(|| Integer::zero(), |a, b| a + b)
            + non_negative_integral_power(integer_two, Self::FLOAT_DIGITS.clone() - Int::one())
                .expect(&format!(
                    concat!(
                        "Error[RealFloat::decode_float]: ",
                        "Failed to compute pow(2, {})"
                    ),
                    Self::FLOAT_DIGITS + Int::one()
                ));
        significand.sign = sign;
        (significand, exponent)
    }

    /// Encode the given significand and exponent into a floating-point number.
    fn encode_float(significand: Integer, exponent: Int) -> Self {
        integral_power(from_integral(Self::FLOAT_RADIX), exponent)
            .map(|p: Self| p * Self::from_integer(significand))
            .expect(concat!(
                "Error[RealFloat::encode_float]: ",
                "Should be able to produce the correct result."
            ))
    }

    /// Return the actual exponent in the floating-point representation.
    ///
    /// In Haskell, it defines like:
    ///
    /// ```haskell
    /// exponent 0 = 0
    /// exponent x = snd (decodeFloat x) + floatDigits x
    /// ```
    fn exponent(self) -> Int {
        if self.is_zero() {
            Int::zero()
        } else {
            self.decode_float().1 + Self::FLOAT_DIGITS
        }
    }

    /// Return the actual significand in the floating-point representation.
    fn significand(self) -> Self {
        Self::encode_float(self.decode_float().0, -Self::FLOAT_DIGITS)
    }

    /// Multiplies a real floating-point number by an integer power of the radix.
    fn scale_float(self, factor: Int) -> Self {
        if self.is_zero() || self.is_not_a_number() || self.is_infinite_number() {
            self
        } else {
            let (significand, exponent) = self.decode_float();
            let (lower_boundary, upper_boundary) = Self::FLOAT_RANGE;
            let factor_p = upper_boundary - lower_boundary + Int::of(4) * Self::FLOAT_DIGITS;
            Self::encode_float(significand, exponent + clamp(factor_p, factor))
        }
    }

    /// Validate whether a given real floating-point number is NaN.
    fn is_not_a_number(&self) -> bool;

    /// Validate whether a given real floating-point number is Inf.
    fn is_infinite_number(&self) -> bool;

    /// Validate whether a given real floating-point number is in a denormalized form.
    fn is_denormalized(&self) -> bool;

    /// Validate whether a given real floating-point number is "negative zero".
    fn is_negative_zero(&self) -> bool;

    /// The two-parameter arctangent function, atan2(y, x).
    fn arc_tangent_2(y: Self, x: Self) -> Self {
        if x > Self::zero() {
            (y / x).arc_tangent()
        } else if x.is_zero() && y > Self::zero() {
            Self::PI * Self::half()
        } else if x < Self::zero() && y > Self::zero() {
            Self::PI + (y / x).arc_tangent()
        } else if (x <= Self::zero() && y < Self::zero())
            || (x < Self::zero() && y.is_negative_zero())
            || (x.is_negative_zero() && y.is_negative_zero())
        {
            -Self::arc_tangent_2(-y, x)
        } else if y.is_zero() && (x < Self::zero() || x.is_negative_zero()) {
            Self::PI
        } else if x.is_zero() && y.is_zero() {
            y
        } else {
            x + y
        }
    }

    /// sqrt(x^2 + y^2)
    fn hypot(self, rhs: Self) -> Self;
}
