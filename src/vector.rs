//! vector operations
//!
//! a matrix is said to be a vector,
//! if the number of columns of the matrix is 0
//!
//! the default vector is the column vector,
//! and the row vector is the transpose of the column vector

use crate::{number::INum, Matrix};

pub type Vector<N, const R: usize> = Matrix<N, R, 1>;

pub trait IVec {
    /// vector element type
    type Val;

    fn euclidean_norm(&self) -> f64;

    fn unit(&self) -> Self;

    fn inner_product(&self, rhs: &Self) -> Self::Val;

    fn outer_product(&self, rhs: &Self) -> Self;

    fn project(&self, rhs: &Self) -> Self;
}

impl<'de, N: INum<'de>, const R: usize> IVec for Vector<N, R> {
    type Val = N;

    fn euclidean_norm(&self) -> f64 {
        let mut result = N::default();
        for i in 1..=R {
            result = result + self.get(i, 1).clone() * self.get(i, 1).clone();
        }
        let result: f64 = result.into();
        result.sqrt()
    }

    fn unit(&self) -> Self {
        assert!(!self.is_zero());
        let u = self.clone();
        u.smul(N::one() / N::from_f64(u.euclidean_norm()))
    }

    fn inner_product(&self, rhs: &Self) -> Self::Val {
        let mut s = N::default();
        for i in 1..=R {
            s = s + self.get(i, 1).clone() * rhs.get(i, 1).clone();
        }
        s
    }

    fn outer_product(&self, rhs: &Self) -> Self {
        assert!(3 == R);
        let mut z = Matrix::zeros();
        z.set(
            1,
            1,
            self.get(2, 1).clone() * rhs.get(3, 1).clone()
                - self.get(3, 1).clone() * rhs.get(2, 1).clone(),
        );
        z.set(
            2,
            1,
            self.get(3, 1).clone() * rhs.get(1, 1).clone()
                - self.get(1, 1).clone() * rhs.get(3, 1).clone(),
        );
        z.set(
            3,
            1,
            self.get(1, 1).clone() * rhs.get(2, 1).clone()
                - self.get(2, 1).clone() * rhs.get(1, 1).clone(),
        );
        z
    }

    fn project(&self, rhs: &Self) -> Self {
        let rhs_length = rhs.euclidean_norm();
        rhs.smul(self.inner_product(rhs) / N::from_f64(rhs_length.clone() * rhs_length))
    }
}
