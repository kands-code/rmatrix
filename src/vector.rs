//! # Vector
//!
//! vector manipulations

use crate::matrix::Matrix;
use crate::number::Number;
use crate::{error::Result, number::Fractional};

#[cfg(feature = "rayon_mat")]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

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
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let vec1: VectorC<f32, 3> = VectorC::create(vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: VectorC<f32, 3> = VectorC::create(vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(VectorC::create(vec![-3.0f32, 6.0f32, -3.0f32])?, times_c(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn times_c<T>(vec1: VectorC<T, 3>, vec2: VectorC<T, 3>) -> Result<VectorC<T, 3>>
where
    T: Number,
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
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let vec1: VectorC<f32, 3> = VectorC::create(vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: VectorC<f32, 3> = VectorC::create(vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(32.0f32, times_d(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn times_d<T, const ROW: usize>(vec1: VectorC<T, ROW>, vec2: VectorC<T, ROW>) -> Result<T>
where
    T: Number,
{
    #[cfg(feature = "rayon_mat")]
    let prod = vec1
        .inner
        .par_iter()
        .zip(vec2.inner.par_iter())
        .map(|(e1, e2)| e1.to_owned() * e2.to_owned())
        .sum();

    #[cfg(not(feature = "rayon_mat"))]
    let prod = vec1
        .inner
        .iter()
        .zip(vec2.inner.iter())
        .map(|(e1, e2)| e1.to_owned() * e2.to_owned())
        .sum();

    Ok(prod)
}

/// vector productor
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::vector::VectorC;
/// # use rmatrix_ks::vector::VectorR;
/// # use rmatrix_ks::vector::times_v;
/// # use rmatrix_ks::error::Error;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let vec1: VectorC<f32, 3> = VectorC::create(vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: VectorR<f32, 3> = VectorR::create(vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(Matrix::create(
///     vec![
///         4.0f32, 5.0f32, 6.0f32,
///         8.0f32, 10.0f32, 12.0f32,
///         12.0f32, 15.0f32, 18.0f32])?,
///     times_v(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn times_v<T, const ROW: usize, const COL: usize>(
    vector_c: VectorC<T, ROW>,
    vector_r: VectorR<T, COL>,
) -> Result<Matrix<T, ROW, COL>>
where
    T: Number,
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

pub fn l2_norm_c<T, const ROW: usize>(vc: VectorC<T, ROW>) -> T
where
    T: Fractional,
{
    #[cfg(feature = "rayon_mat")]
    let norm = vc
        .inner
        .par_iter()
        .map(|e| e.to_owned() * e.to_owned())
        .sum::<T>()
        .sqrt();

    #[cfg(not(feature = "rayon_mat"))]
    let norm = vc
        .inner
        .iter()
        .map(|e| e.to_owned() * e.to_owned())
        .sum::<T>()
        .sqrt();

    norm
}
