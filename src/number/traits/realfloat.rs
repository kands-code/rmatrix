//! # Number Trait :: RealFloat
//!
//! Floating point number for Real.

use crate::number::{
    instances::{int::Int, integer::Integer},
    traits::{floating::Floating, integral::Integral, one::One, realfrac::RealFrac, zero::Zero},
    utils::{clamp, from_integral, integral_power, non_negative_integral_power},
};

/// RealFloat
pub trait RealFloat: RealFrac + Floating {
    /// the radix of the representation
    ///
    /// by default is 2-based
    const FLOAT_RADIX: Int = Int::of(2);

    /// the number of digits of `FLOAT_RADIX` in the significand
    const FLOAT_DIGITS: Int;

    /// the lowest and highest values the exponent may assume
    const FLOAT_RANGE: (Int, Int);

    ///  returns the significand expressed and an appropriately scaled exponent
    ///
    /// ```rust,ignore
    /// use rmatrix_ks::number::utils::integral_power;
    ///
    /// let (significand, exponent) = real_float_number.clone().decode_float();
    /// let radix = F::FLOAT_RADIX;
    /// assert_eq!(real_float_number, significand * integral_power(radix, exponent));
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

    /// inverse of decode_float
    fn encode_float(significand: Integer, exponent: Int) -> Self {
        integral_power(from_integral(Self::FLOAT_RADIX), exponent)
            .map(|p: Self| p * Self::from_integer(significand))
            .expect("Error[RealFloat::encode_float]: Should be able to produce the correct result.")
    }

    /// corresponds to the second component of decode_float
    ///
    /// In Haskell, it defines like:
    ///
    /// ```haskell
    /// exponent 0 = 0
    /// exponent x = snd (decodeFloat x) + floatDigits x
    /// ```
    fn exponent(self) -> Int {
        self.decode_float().1 + Self::FLOAT_DIGITS
    }

    /// corresponds to the first component of decode_float
    fn significand(self) -> Self {
        Self::encode_float(self.decode_float().0, -Self::FLOAT_DIGITS)
    }

    /// multiplies a floating-point number by an integer power of the radix
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

    /// check if it is NaN
    fn is_not_a_number(&self) -> bool;

    /// check if it is Inf
    fn is_infinite_number(&self) -> bool;

    /// check if it is denormalized
    fn is_denormalized(&self) -> bool;

    /// check if it is (-0)
    fn is_negative_zero(&self) -> bool;

    /// atan2(y, x)
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
