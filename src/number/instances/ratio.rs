//! # Number Type :: Ratio
//!
//! For any Integral type number x and y,
//! a Ratio type based on this type is x / y.

use crate::number::{
    instances::integer::Integer,
    traits::{
        fractional::Fractional, integral::Integral, number::Number, one::One, real::Real,
        realfrac::RealFrac, zero::Zero,
    },
    utils::{from_integral, gcd},
};

#[derive(Clone)]
pub struct Ratio<I: Integral> {
    pub numerator: I,
    pub denominator: I,
}

pub type Rational = Ratio<Integer>;

impl<I: Integral> Ratio<I> {
    pub const fn of(numerator: I, denominator: I) -> Self {
        Ratio {
            numerator,
            denominator,
        }
    }

    // create Ratio from &str
    pub fn of_str(ratio_numbwe: &str) -> Option<Self> {
        std::str::FromStr::from_str(ratio_numbwe).ok()
    }

    pub fn refine(self) -> Self {
        if self.is_zero() {
            Self::zero()
        } else {
            let sign_number = if self.numerator.sign_number() == self.denominator.sign_number() {
                I::one()
            } else {
                -I::one()
            };
            if self.numerator.absolute_value() == self.denominator.absolute_value() {
                Self::of(sign_number, I::one())
            } else {
                let common = gcd(self.numerator.clone(), self.denominator.clone());
                Self::of(
                    self.numerator.quotient(common.clone()).absolute_value() * sign_number,
                    self.denominator.quotient(common).absolute_value(),
                )
            }
        }
    }
}

impl<I: Integral> One for Ratio<I> {
    fn one() -> Self {
        Self {
            numerator: I::one(),
            denominator: I::one(),
        }
    }

    fn is_one(&self) -> bool {
        let cloned = self.clone();
        gcd(cloned.numerator, cloned.denominator).is_one()
    }
}
impl<I: Integral> Zero for Ratio<I> {
    fn zero() -> Self {
        Self {
            numerator: I::zero(),
            denominator: I::one(),
        }
    }

    fn is_zero(&self) -> bool {
        self.numerator.is_zero() && !self.denominator.is_zero()
    }
}

impl<I: Integral> std::default::Default for Ratio<I> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<I: Integral> std::cmp::PartialEq for Ratio<I> {
    fn eq(&self, rhs: &Self) -> bool {
        let refined_lhs = self.clone().refine();
        let refined_rhs = rhs.clone().refine();
        refined_lhs.numerator == refined_rhs.numerator
            && refined_lhs.denominator == refined_rhs.denominator
    }
}

impl<I: Integral> std::cmp::PartialOrd for Ratio<I> {
    fn partial_cmp(&self, rhs: &Self) -> Option<std::cmp::Ordering> {
        if self.denominator.is_zero() || rhs.denominator.is_zero() {
            None
        } else {
            let difference = self.clone() - rhs.clone();
            Some(if difference.numerator > I::zero() {
                std::cmp::Ordering::Greater
            } else if difference.numerator.is_zero() {
                std::cmp::Ordering::Equal
            } else {
                std::cmp::Ordering::Less
            })
        }
    }
}

impl<I: Integral> std::ops::Neg for Ratio<I> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let refined = self.refine();
        Self::of(-refined.numerator, refined.denominator)
    }
}

impl<I: Integral> std::ops::Add for Ratio<I> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let refined_lhs = self.refine();
        let refined_rhs = rhs.refine();
        let lhs_numerator = refined_lhs.numerator.to_integer();
        let lhs_denominator = refined_lhs.denominator.to_integer();
        let rhs_numerator = refined_rhs.numerator.to_integer();
        let rhs_denominator = refined_rhs.denominator.to_integer();
        let numerator =
            lhs_numerator * rhs_denominator.clone() + rhs_numerator * lhs_denominator.clone();
        let denominator = lhs_denominator * rhs_denominator;
        let ratio_sum = Rational::of(numerator, denominator).refine();
        Self::of(
            I::from_integer(ratio_sum.numerator),
            I::from_integer(ratio_sum.denominator),
        )
    }
}

impl<I: Integral> std::ops::Sub for Ratio<I> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<I: Integral> std::ops::Mul for Ratio<I> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        // refine
        let refined_lhs = self.refine();
        let refined_rhs = rhs.refine();
        // gcd
        let lhs_common = gcd(
            refined_lhs.numerator.clone(),
            refined_rhs.denominator.clone(),
        );
        let rhs_common = gcd(
            refined_rhs.numerator.clone(),
            refined_lhs.denominator.clone(),
        );
        // clear
        let refined_lhs_numerator = refined_lhs.numerator.quotient(lhs_common.clone());
        let refined_lhs_denominator = refined_lhs.denominator.quotient(rhs_common.clone());
        let refined_rhs_numerator = refined_rhs.numerator.quotient(rhs_common);
        let refined_rhs_denominator = refined_rhs.denominator.quotient(lhs_common);
        // mult
        let numerator = refined_lhs_numerator * refined_rhs_numerator;
        let denominator = refined_lhs_denominator * refined_rhs_denominator;
        Self::of(numerator, denominator).refine()
    }
}

impl<I: Integral> std::ops::Div for Ratio<I> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.reciprocal()
    }
}

impl<I: Integral> Number for Ratio<I> {
    fn absolute_value(&self) -> Self {
        let refined = self.clone().refine();
        Self::of(refined.numerator.absolute_value(), refined.denominator)
    }

    fn sign_number(&self) -> Self {
        let refined = self.clone().refine();
        if refined.numerator > I::zero() {
            Self::one()
        } else if refined.numerator.is_zero() {
            Self::zero()
        } else {
            Self {
                numerator: -I::one(),
                denominator: I::one(),
            }
        }
    }

    fn from_integer(integer_number: Integer) -> Self {
        Self::of(I::from_integer(integer_number), I::one())
    }
}

impl<I: Integral> Fractional for Ratio<I> {
    fn half() -> Self {
        Ratio::of(I::one(), I::one() + I::one())
    }

    fn reciprocal(self) -> Self {
        Self::of(self.denominator, self.numerator)
    }

    fn from_rational(rational_number: Rational) -> Self {
        let refined = rational_number.refine();
        Self::of(
            I::from_integer(refined.numerator),
            I::from_integer(refined.denominator),
        )
    }
}

impl<I: Integral> Real for Ratio<I> {
    fn to_rational(self) -> Rational {
        let refined = self.refine();
        Rational::of(
            refined.numerator.to_integer(),
            refined.denominator.to_integer(),
        )
    }
}

impl<I: Integral> RealFrac for Ratio<I> {
    fn proper_fraction<II: Integral>(self) -> (II, Self) {
        let refined = self.refine();
        if refined.numerator < refined.denominator {
            (II::zero(), refined)
        } else {
            let (quot, rem) = refined.numerator.quot_rem(refined.denominator.clone());
            (from_integral(quot), Self::of(rem, refined.denominator))
        }
    }
}

impl<I: Integral> std::fmt::Debug for Ratio<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:+} % {:+}", self.numerator, self.denominator)
    }
}

impl<I: Integral> std::fmt::Display for Ratio<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} % {}", self.numerator, self.denominator)
    }
}

impl<I: Integral> std::str::FromStr for Ratio<I> {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // remove extra whitespaces
        let trimmed_s = s.trim();
        // use regular expressions to validate string
        let searcher = regex::Regex::new(
            r"(?<numerator>([+-]?)([0-9_]+))([\s]*[%][\s]*)(?<denominator>([+-]?)([0-9_]+))",
        )
        .expect("Error[Ratio::from_str]: Should be a valid regular expression.");
        if let Some(captures) = searcher.captures(trimmed_s) {
            let numerator_s = &captures["numerator"];
            let denominator_s = &captures["denominator"];
            let numerator = std::str::FromStr::from_str(numerator_s);
            let denominator = std::str::FromStr::from_str(denominator_s);
            match (numerator, denominator) {
                (Ok(n), Ok(d)) => Ok(Self::of(n, d)),
                _ => {
                    eprintln!(
                        "Error[Ratio::from_str]: ({}) or ({}) is not a valid Integral literal.",
                        numerator_s, denominator_s
                    );
                    Err(())
                }
            }
        } else {
            eprintln!(
                "Error[Ratio::from_str]: ({}) is not a valid Ratio literal.",
                trimmed_s
            );
            Err(())
        }
    }
}
