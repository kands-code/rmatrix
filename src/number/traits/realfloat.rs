//! # traits::realfloat
//!
//! Types that implement this trait can be considered as real floating-point numbers.

use crate::number::{
    instances::{int::Int, integer::Integer},
    traits::{floating::Floating, integral::Integral, one::One, realfrac::RealFrac, zero::Zero},
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

    ///  Decode a real floating-point number into its significand and exponent.
    ///
    /// **NEED FIX**
    ///
    /// ```rust,ignore
    /// let rfp : F;
    /// let (significand, exponent) = rfp.decode_float();
    /// let radix = F::FLOAT_RADIX;
    /// assert_eq!(significand * integral_power(radix, exponent), rfp);
    /// ```
    fn decode_float(self) -> (Integer, Int) {
        let range = Self::FLOAT_RANGE.1 + Int::one();
        let exponent = self
            .absolute_value()
            .logarithmic_base(from_integral(Self::FLOAT_RADIX))
            .ceiling::<Int>();
        let modified_exponent = if self.is_zero() {
            Int::zero()
        } else if exponent.is_zero() {
            range - Self::FLOAT_DIGITS
        } else {
            exponent.clone() - Self::FLOAT_DIGITS
        };
        let significand =
            integral_power(from_integral(Self::FLOAT_RADIX), modified_exponent.clone())
                .map(|p| self.clone() / p)
                .map(|p| p.to_rational().numerator)
                .expect(
                    "Error[RealFloat::decode_float]: Should be able to produce the correct result.",
                );
        let significand_range =
            non_negative_integral_power(Self::FLOAT_RADIX, Self::FLOAT_DIGITS + Int::one())
                .expect(
                    "Error[RealFloat::decode_float]: Should be able to produce the correct result.",
                )
                .to_integer();
        let modified_significand = if modified_exponent.is_zero() {
            significand.modulus(
                if self > Self::zero() {
                    significand_range
                } else {
                    -significand_range
                } - Integer::one(),
            )
        } else {
            significand
        };
        (modified_significand, modified_exponent)
    }

    /// Encode the given significand and exponent into a floating-point number.
    fn encode_float(significand: Integer, exponent: Int) -> Self {
        integral_power(from_integral(Self::FLOAT_RADIX), exponent)
            .map(|p: Self| p * Self::from_integer(significand))
            .expect("Error[RealFloat::encode_float]: Should be able to produce the correct result.")
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
    ///
    /// **NEED FIX**
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
