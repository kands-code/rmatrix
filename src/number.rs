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
    fn is_zero(&self) -> bool;
}

impl Number for f64 {
    fn one() -> Self {
        1.0f64
    }

    fn is_zero(&self) -> bool {
        ((self.abs() * (10.0f64).powi(8)).trunc() as i32).is_zero()
    }
}

impl Number for i32 {
    fn one() -> Self {
        1i32
    }
    fn is_zero(&self) -> bool {
        self == &0i32
    }
}

impl Number for Complex {
    fn one() -> Self {
        Complex::new(f64::one(), Default::default())
    }
    fn is_zero(&self) -> bool {
        self.norm().is_zero()
    }
}
