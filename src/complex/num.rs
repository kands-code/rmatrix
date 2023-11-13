use crate::{complex::Complex, error::RMatrixError};

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
    type Output = Result<Self, RMatrixError>;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            Err(RMatrixError::DivideByZero)
        } else {
            let rhs_norm = rhs.norm();
            Ok(Complex::new(self.re() / rhs_norm, self.im() / rhs_norm))
        }
    }
}

impl std::ops::Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Complex::new(-self.re(), -self.im())
    }
}
