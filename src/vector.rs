//! # Vector
//!
//! vector manipulations

use crate::error::MatrixError;
use crate::matrix::Matrix;

/// vector is a one-dimensional matrix
///
/// by default, the vector is a column vector
pub type VectorC<T, const ROW: usize> = Matrix<T, ROW, 1>;

/// the row vector
pub type VectorR<T, const COL: usize> = Matrix<T, 1, COL>;

pub fn times_v<T, const ROW: usize, const COL: usize>(
    vector_c: VectorC<T, ROW>,
    vector_r: VectorR<T, COL>,
) -> Result<Matrix<T, ROW, COL>, MatrixError>
where
    T: Clone + Default + std::ops::Mul<Output = T>,
{
    let mut mat = Matrix::<T, ROW, COL>::create(vec![T::default(); ROW * COL])?;
    for r in 1..=ROW {
        for c in 1..=COL {
            mat.set_element(
                r,
                c,
                vector_c.get_element(r, 1)?.to_owned() * vector_r.get_element(1, c)?.to_owned(),
            )?;
        }
    }
    Ok(mat)
}
