use crate::complex::Complex;

pub trait Number:
    std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div
    + std::ops::Neg<Output = Self>
    + PartialEq
    + Default
    + Clone
    + Copy
    + std::fmt::Debug
    + std::fmt::Display
    + std::str::FromStr
    + std::iter::Sum
{
    fn one() -> Self;
}

impl Number for f64 {
    fn one() -> Self {
        1.0f64
    }
}

impl Number for i32 {
    fn one() -> Self {
        1i32
    }
}

impl Number for Complex {
    fn one() -> Self {
        Complex::new(f64::one(), Default::default())
    }
}
