//! implementation of Number for Complex

use crate::complex::Complex;

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Complex::new(self.re() + rhs.re(), self.im() + rhs.re())
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Complex::new(self.re() - rhs.re(), self.im() - self.im())
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Complex::new(
            self.re() * rhs.re() - self.im() * rhs.im(),
            self.re() * rhs.im() + self.im() * rhs.re(),
        )
    }
}

impl std::ops::Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Complex::new(self.re() / rhs.norm(), self.im() / rhs.norm())
    }
}

impl std::ops::Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Complex::new(-self.re(), -self.im())
    }
}
