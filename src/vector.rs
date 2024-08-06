//! # Vector
//!
//! vector manipulations

use crate::error::Result;
use crate::matrix::Matrix;
use crate::num::number::Fractional;
use crate::num::number::Number;

/// vector is a one-dimensional matrix
///
/// by default, the vector is a column vector
pub type VectorC<T, const ROW: usize> = Matrix<T, ROW, 1>;
/// the row vector
pub type VectorR<T, const COL: usize> = Matrix<T, 1, COL>;

/// vector cross product
///
/// ```rust
/// # use rmatrix_ks::vector::VectorC;
/// # use rmatrix_ks::vector::times_c;
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

/// vector dot product
///
/// ```rust
/// # use rmatrix_ks::vector::VectorC;
/// # use rmatrix_ks::vector::VectorR;
/// # use rmatrix_ks::vector::times_d;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let vec1: VectorR<f32, 3> = VectorR::create(vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: VectorC<f32, 3> = VectorC::create(vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(32.0f32, times_d(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn times_d<T, const EDGE: usize>(vec1: VectorR<T, EDGE>, vec2: VectorC<T, EDGE>) -> Result<T>
where
    T: Number,
{
    Ok(vec1
        .inner
        .iter()
        .zip(vec2.inner.iter())
        .map(|(e1, e2)| e1.to_owned() * e2.to_owned())
        .sum())
}

/// vector convolution
///
/// ```rust
/// # use rmatrix_ks::vector::VectorC;
/// # use rmatrix_ks::vector::convolution;
/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let vec1: VectorC<f32, 3> = VectorC::create(vec![1.0f32, 2.0f32, 3.0f32])?;
/// let vec2: VectorC<f32, 3> = VectorC::create(vec![4.0f32, 5.0f32, 6.0f32])?;
/// assert_eq!(VectorC::create(vec![4.0f32, 13.0f32, 28.0f32, 27.0f32, 18.0f32])?,
///     convolution(vec1, vec2)?);
/// # Ok(())
/// # }
/// ```
pub fn convolution<T, const M: usize, const N: usize>(
    vec1: VectorC<T, M>,
    vec2: VectorC<T, N>,
) -> Result<VectorC<T, { M + N - 1 }>>
where
    T: Number,
{
    let mut conv: VectorC<T, { M + N - 1 }> = VectorC::zeros()?;

    for k in 1..=(N + M - 1) {
        for i in 1..=(k.min(M)) {
            // k + 1 = i + j
            let j = k + 1 - i;
            if (1..=N).contains(&j) {
                conv.set_element(
                    k,
                    1,
                    conv.get_element(k, 1)?.to_owned()
                        + vec1.get_element(i, 1)?.to_owned() * vec2.get_element(j, 1)?.to_owned(),
                )?;
            }
        }
    }

    Ok(conv)
}

/// vector product
///
/// ```rust
/// # use rmatrix_ks::matrix::Matrix;
/// # use rmatrix_ks::vector::VectorC;
/// # use rmatrix_ks::vector::VectorR;
/// # use rmatrix_ks::vector::times_v;

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

/// euclid norm for vector
///
/// aka l2-norm
///
/// ```rust
/// # use rmatrix_ks::vector::VectorC;
/// # use rmatrix_ks::vector::euclid_norm;

/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let v: VectorC<f32, 2> = VectorC::create(vec![3.0f32, 4.0f32])?;
/// assert_eq!(5.0f32, euclid_norm(v)?);
/// # Ok(())
/// # }
/// ```
pub fn euclid_norm<T, const ROW: usize>(vc: VectorC<T, ROW>) -> Result<T>
where
    T: Fractional,
{
    Ok(times_d(vc.conjugate_transpose()?, vc)?.sqrt())
}

/// root mean square for a vector
///
/// ```rust
/// # use rmatrix_ks::vector::VectorC;
/// # use rmatrix_ks::vector::root_mean_square;

/// # use rmatrix_ks::error::Result;
/// # fn main() -> Result<()> {
/// let v: VectorC<f32, 2> = VectorC::create(vec![3.0f32, 4.0f32])?;
/// assert_eq!(5.0f32 / 2.0f32.sqrt(), root_mean_square(v)?);
/// # Ok(())
/// # }
/// ```
pub fn root_mean_square<T, const ROW: usize>(vc: VectorC<T, ROW>) -> Result<T>
where
    T: Fractional,
{
    euclid_norm(vc)?.ndiv(T::from_usize(ROW).sqrt())
}
