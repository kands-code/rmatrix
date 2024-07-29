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

/// vector cross productor
///
/// ```rust
/// # use rmatrix_ks::vector::VectorC;
/// # use rmatrix_ks::vector::times_c;
/// # use rmatrix_ks::error::MatrixError;
/// # fn main() -> Result<(), MatrixError> {
/// let vec1: VectorC<i8, 3> = VectorC::create(vec![1, 2, 3])?;
/// let vec2: VectorC<i8, 3> = VectorC::create(vec![4, 5, 6])?;
/// assert_eq!(VectorC::create(vec![-3, 6, -3])?, times_c(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn times_c<T>(vec1: VectorC<T, 3>, vec2: VectorC<T, 3>) -> Result<VectorC<T, 3>, MatrixError>
where
    T: Clone + std::ops::Sub<Output = T> + std::ops::Mul<Output = T>,
{
    VectorC::create(vec![
        vec1.get_element(2, 1)?.to_owned() * vec2.get_element(3, 1)?.to_owned()
            - vec1.get_element(3, 1)?.to_owned() * vec2.get_element(2, 1)?.to_owned(),
        vec1.get_element(3, 1)?.to_owned() * vec2.get_element(1, 1)?.to_owned()
            - vec1.get_element(1, 1)?.to_owned() * vec2.get_element(3, 1)?.to_owned(),
        vec1.get_element(1, 1)?.to_owned() * vec2.get_element(2, 1)?.to_owned()
            - vec1.get_element(2, 1)?.to_owned() * vec2.get_element(1, 1)?.to_owned(),
    ])
}

/// vector dot productor
///
/// ```rust
/// # use rmatrix_ks::vector::VectorC;
/// # use rmatrix_ks::vector::times_d;
/// # use rmatrix_ks::error::MatrixError;
/// # fn main() -> Result<(), MatrixError> {
/// let vec1: VectorC<i8, 3> = VectorC::create(vec![1, 2, 3])?;
/// let vec2: VectorC<i8, 3> = VectorC::create(vec![4, 5, 6])?;
/// assert_eq!(32, times_d(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn times_d<T, const ROW: usize>(
    vec1: VectorC<T, ROW>,
    vec2: VectorC<T, ROW>,
) -> Result<T, MatrixError>
where
    T: std::ops::Mul<Output = T> + std::iter::Sum,
{
    Ok(std::iter::zip(vec1.inner, vec2.inner)
        .map(|(e1, e2)| e1 * e2)
        .sum())
}

/// vector productor
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::vector::VectorC;
/// # use rmatrix_ks::vector::VectorR;
/// # use rmatrix_ks::vector::times_v;
/// # use rmatrix_ks::error::MatrixError;
/// # fn main() -> Result<(), MatrixError> {
/// let vec1: VectorC<i8, 3> = VectorC::create(vec![1, 2, 3])?;
/// let vec2: VectorR<i8, 3> = VectorR::create(vec![4, 5, 6])?;
/// assert_eq!(Matrix::create(vec![4, 5, 6, 8, 10, 12, 12, 15, 18])?, times_v(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
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
