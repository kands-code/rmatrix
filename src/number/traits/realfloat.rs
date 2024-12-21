//! # traits::realfloat
//!
//! Types that implement this trait can be considered as real floating-point numbers.

use crate::number::{
    instances::{int::Int, integer::Integer},
    traits::{floating::Floating, realfrac::RealFrac, zero::Zero},
    utils::{clamp, decimal_to_binary, from_integral, integral_power, non_negative_integral_power},
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
        let rfp = self.absolute_value();
        let (exponent_digits, float_digits) = decimal_to_binary(rfp).expect(&format!(
            "Error[RealFloat::decode_float]: Failed to convert ({}) to binary format.",
            self
        ));
        let exponent = Int::of(exponent_digits.len() as i32) - Self::FLOAT_DIGITS;
        let significand_digits = vec![exponent_digits, float_digits].concat();
        let two = Integer::of(true, &[2])
            .expect("Error[RealFloat::decode_float]: Failed to obtain integer two.");
        let mut significand = Integer::zero();
        for (index, &base) in significand_digits.iter().enumerate().rev() {
            if base == 1u8 {
                let exp = significand_digits.len() - index - 1;
                significand = significand
                    + non_negative_integral_power(two.clone(), Int::of(exp as i32)).expect(
                        &format!(
                            "Error[RealFloat::decode_float]: Failed to compute pow(2, {})",
                            exp
                        ),
                    );
            }
        }
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
}
