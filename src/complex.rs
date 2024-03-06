//! simple define of complex number
//!
//! it implements the Num trait, so that the matrix can use this type

use crate::number::INum;
use regex::Regex;
use serde::{ser::SerializeTupleStruct, Deserialize};

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Complex<N>(pub N, pub N);

impl<'de, N: INum<'de>> Complex<N> {
    /// get the real part of a complex number
    ///
    /// ```rust
    /// # use rmatrix_ks::complex::Complex;
    /// # fn main() {
    /// let c = Complex(3, 4);
    /// assert_eq!(c.real(), 3);
    /// # }
    /// ```
    pub fn real(&self) -> N {
        self.0.clone()
    }

    /// get the imagine part of a complex number
    ///
    /// ```rust
    /// # use rmatrix_ks::complex::Complex;
    /// # fn main() {
    /// let c = Complex(3, 4);
    /// assert_eq!(c.imag(), 4);
    /// # }
    /// ```
    pub fn imag(&self) -> N {
        self.1.clone()
    }

    /// the base norm of a complex number
    ///
    /// ```rust
    /// # use rmatrix_ks::complex::Complex;
    /// # fn main() {
    /// assert_eq!(Complex(3, -4).base_norm(), 25);
    /// # }
    /// ```
    pub fn base_norm(&self) -> N {
        self.real() * self.real() + self.imag() * self.imag()
    }

    /// the norm of a complex number
    ///
    /// ```rust
    /// # use rmatrix_ks::complex::Complex;
    /// # fn main() {
    /// assert_eq!(Complex(3, -4).norm(), 5.0_f64);
    /// # }
    /// ```
    pub fn norm(&self) -> f64 {
        let bn: f64 = self.base_norm().into();
        bn.sqrt()
    }
}

impl<'de, N: INum<'de>> INum<'de> for Complex<N> {
    fn one() -> Self {
        Complex(N::one(), N::default())
    }

    fn is_zero(&self) -> bool {
        self.real().is_zero() && self.imag().is_zero()
    }

    fn from_f64(value: f64) -> Self {
        Complex(N::from_f64(value), N::default())
    }
}

impl<'de, N: INum<'de>> std::ops::Add for Complex<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Complex(self.real() + rhs.real(), self.imag() + rhs.imag())
    }
}

impl<'de, N: INum<'de>> std::ops::Sub for Complex<N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Complex(self.real() - rhs.real(), self.imag() - rhs.imag())
    }
}

impl<'de, N: INum<'de>> std::ops::Mul for Complex<N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Complex(
            self.real() * rhs.real() - self.imag() * rhs.imag(),
            self.real() * rhs.imag() + self.imag() * rhs.real(),
        )
    }
}

impl<'de, N: INum<'de>> std::ops::Div for Complex<N> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        assert!(!rhs.is_zero());

        Complex(
            (self.real() * rhs.real() + self.imag() * rhs.imag()) / rhs.base_norm(),
            (self.imag() * rhs.real() - self.real() * rhs.imag()) / rhs.base_norm(),
        )
    }
}

impl<'de, N: INum<'de>> std::ops::Neg for Complex<N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Complex(self.real().neg(), self.imag().neg())
    }
}

impl<'de, N: INum<'de>> Default for Complex<N> {
    fn default() -> Self {
        Complex(N::default(), N::default())
    }
}

impl<'de, N: INum<'de>> std::fmt::Display for Complex<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{:+}I", self.real(), self.imag())
    }
}

/// a legal complex number string looks like "3.2 - 10I"
///
/// ```rust
/// # use rmatrix_ks::complex::Complex;
/// # use rmatrix_ks::number::INum;
/// use std::str::FromStr;
/// # fn main() {
/// let s = "3.2 - 10I";
/// let c: Complex<f32> = Complex::from_str(s).unwrap();
/// assert!((c - Complex(3.2_f32, -10.0_f32)).is_zero())
/// # }
/// ```
impl<'de, N: INum<'de>> std::str::FromStr for Complex<N> {
    type Err = Box<dyn std::error::Error>;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let complex_regex: &str = r"(?<re>[+-]?[0-9]+\.?[0-9]*)(?<im>[+-]?[0-9]+\.?[0-9]*)I$";
        let regex_engine = Regex::new(complex_regex)?;

        let parse_str = s.replace(|c: char| c.is_whitespace(), "");
        let cap = regex_engine.captures(&parse_str).unwrap();
        Ok(Complex(
            (&cap["re"]).parse().unwrap_or_default(),
            (&cap["im"]).parse().unwrap_or_default(),
        ))
    }
}

impl<'de, N: INum<'de>> std::iter::Sum for Complex<N> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), move |a, b| a + b)
    }
}

/// convert a complex number to a real number
///
/// only keep the real part of the complex number
///
/// ```rust
/// # use rmatrix_ks::complex::Complex;
/// # use rmatrix_ks::number::INum;
/// # fn main() {
/// assert_eq!(3.0_f64, <Complex<f64> as Into<f64>>::into(Complex(3.0_f64, 4.0_f64)));
/// # }
/// ```
impl<'de, N: INum<'de>> std::convert::Into<f64> for Complex<N> {
    fn into(self) -> f64 {
        self.real().into()
    }
}

impl<'de, N: INum<'de>> serde::Serialize for Complex<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut s = serializer.serialize_tuple_struct("Complex", 2).unwrap();
        s.serialize_field(&self.0)?;
        s.serialize_field(&self.1)?;
        s.end()
    }
}
