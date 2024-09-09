//! # Complex
//!
//! complex number
//!
//! just an example

use crate::error::IError;
use crate::error::IResult;
use crate::num::number::IFractional;
use crate::num::number::INumber;
use crate::num::number::IOne;
use crate::num::number::IZero;

use super::number::IEqual;

/// complex number
#[derive(std::clone::Clone, std::cmp::PartialEq, std::fmt::Debug)]
#[cfg_attr(feature = "serde_mat", derive(serde::Deserialize, serde::Serialize))]
pub struct Complex<T: INumber> {
    pub real: T,
    pub imag: T,
}

impl<T: INumber> Complex<T> {
    /// create a complex number
    ///
    /// ```rust
    /// # use rmatrix_ks::cmplx;
    /// # use rmatrix_ks::num::complex::Complex;
    /// # fn main() {
    /// let _ = Complex::create(1i32, 2i32); // 1 + 2I
    /// // or can use cmplx!()
    /// let _ = cmplx!(3i32, 4i32); // 3 + 4I
    /// # }
    /// ```
    pub fn create(real: T, imag: T) -> Self {
        Complex { real, imag }
    }

    /// norm for complex number
    ///
    /// norm(a + bI) = sqrt(a^2 + b^2)
    ///
    /// ```rust
    /// # use rmatrix_ks::error::IResult;
    /// # use rmatrix_ks::num::complex::Complex;
    /// # fn main() -> IResult<()> {
    /// let c = Complex::create(3.0f32, 4.0f32); // 3 + 4I
    /// assert_eq!(5.0f32, c.norm()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm(&self) -> IResult<T>
    where
        T: IFractional,
    {
        (self.real.clone() * self.real.clone() + self.imag.clone() * self.imag.clone()).nsqrt()
    }
}

#[macro_export]
/// equivalent to Complex::create()
macro_rules! cmplx {
    ($x:expr, $y:expr) => {
        Complex::create($x, $y)
    };
}

impl<T: INumber> std::fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<{}, {}>", self.real, self.imag)
    }
}

impl<T: INumber> std::ops::Neg for Complex<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::create(-self.real, -self.imag)
    }
}

impl<T: INumber> std::ops::Add for Complex<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::create(self.real + rhs.real, self.imag + rhs.imag)
    }
}

impl<T: INumber> std::iter::Sum for Complex<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, e| acc + e)
    }
}

impl<T: INumber> std::ops::Sub for Complex<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::create(self.real - rhs.real, self.imag - rhs.imag)
    }
}

impl<T: INumber> std::ops::Mul for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::create(
            self.real.clone() * rhs.real.clone() - self.imag.clone() * rhs.imag.clone(),
            self.real * rhs.imag + self.imag * rhs.real,
        )
    }
}

impl<T: INumber> Default for Complex<T> {
    fn default() -> Self {
        Self {
            real: T::zero(),
            imag: T::zero(),
        }
    }
}

impl<T: INumber> IEqual for Complex<T> {
    fn equal(&self, rhs: &Self) -> bool {
        self.real.equal(&rhs.real) && self.imag.equal(&rhs.imag)
    }
}

impl<T: INumber> IZero for Complex<T> {
    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.imag.is_zero()
    }
}

impl<T: INumber> IOne for Complex<T> {
    fn one() -> Self {
        Self::create(T::one(), T::zero())
    }

    fn is_one(&self) -> bool {
        self.real.is_one() && self.imag.is_zero()
    }
}

/// complex number is a  number
impl<T: INumber> INumber for Complex<T> {
    /// absolute value for complex number
    ///
    /// not the norm of complex number
    fn abs(self) -> Self {
        Self::create(self.real.abs(), self.imag.abs())
    }

    /// conjugate for complex number
    ///
    /// conj(a + bI) = a - bI
    ///
    /// ```rust
    /// # use rmatrix_ks::cmplx;
    /// # use rmatrix_ks::num::complex::Complex;
    /// # use rmatrix_ks::num::number::INumber;
    /// # fn main() {
    /// let c = Complex::create(3.0f32, 4.0f32); // 3 + 4I
    /// assert_eq!(cmplx!(3.0f32, -4.0f32), c.conjugate());
    /// # }
    /// ```
    fn conjugate(self) -> Self {
        Complex::create(self.real, -self.imag)
    }

    fn ndiv(self, rhs: Self) -> IResult<Self> {
        if rhs.is_zero() {
            Err(IError::DividedByZero)
        } else {
            let under = rhs.real.clone() * rhs.real.clone() + rhs.imag.clone() * rhs.imag.clone();
            Ok(Complex::create(
                (self.real.clone() * rhs.real.clone() + self.imag.clone() * rhs.imag.clone())
                    .ndiv(under.clone())?,
                (self.imag * rhs.real - self.real * rhs.imag).ndiv(under)?,
            ))
        }
    }
}

impl<T: IFractional> IFractional for Complex<T> {
    /// one of the numeric value of sqrt(c)
    ///
    /// ```rust
    /// # use rmatrix_ks::error::IResult;
    /// # use rmatrix_ks::num::complex::Complex;
    /// # use crate::rmatrix_ks::num::number::IFractional;
    /// # fn main() -> IResult<()> {
    /// let c = Complex::create(3.0f32, 4.0f32); // 3 + 4I
    /// // sqrt(3 + 4I) = 2 + 1I
    /// assert_eq!(Complex::create(2.0f32, 1.0f32), c.nsqrt()?);
    /// # Ok(())
    /// # }
    /// ```
    fn nsqrt(self) -> IResult<Self> {
        if self.imag.is_zero() {
            Ok(Complex::create(self.real.nsqrt()?, T::zero()))
        } else {
            let real = self.real;
            let imag = self.imag;
            let sqrt_two = (T::one() + T::one()).nsqrt()?;
            let aux = (real.clone()
                + (real.clone() * real.clone() + imag.clone() * imag.clone()).nsqrt()?)
            .nsqrt()?;
            let sqrt_real = aux.clone().ndiv(sqrt_two.clone())?;
            let sqrt_imag = (-sqrt_two.clone() * real.clone() * aux.clone()
                + (aux.clone() * aux.clone() * aux).ndiv(sqrt_two)?)
            .ndiv(imag)?;
            Ok(Complex::create(sqrt_real, sqrt_imag))
        }
    }

    fn from_usize(size: usize) -> Self {
        Complex::create(T::from_usize(size), T::zero())
    }
}
