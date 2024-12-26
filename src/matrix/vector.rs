//! # matrix::vector
//!
//! Definitions of column vectors and row vectors,
//! along with related functions.
//!
//! Default vectors are column vectors,
//! so most functions are implemented only for column vectors.
//! For row vectors, you can first transpose them into column vectors
//! and then use the corresponding functions.

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    matrix::matrix::Matrix,
    number::{
        instances::integer::Integer,
        traits::{number::Number, real::Real, realfloat::RealFloat, realfrac::RealFrac},
    },
};

/// A row vector is a matrix with a single row.
pub type VectorR<N, const C: usize> = Matrix<N, 1, C>;

/// A column vector is a matrix with a single column.
pub type VectorC<N, const R: usize> = Matrix<N, R, 1>;

/// Used to obtain a reference to the element
/// at the corresponding position in the column vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{index_c, VectorC},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let v1: VectorC<Word8, 3> = VectorC::of(&[Word8::of(1), Word8::of(2), Word8::of(3)]).unwrap();
///     assert_eq!(index_c(&v1, 2), &Word8::of(2));
/// }
/// ```
pub fn index_c<N, const R: usize>(v: &VectorC<N, R>, row_index: usize) -> &N {
    &v[(row_index, 1)]
}

/// Used to obtain a reference to the element
/// at the corresponding position in the row vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{index_r, VectorR},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let v1: VectorR<Word8, 3> = VectorR::of(&[Word8::of(1), Word8::of(2), Word8::of(3)]).unwrap();
///     assert_eq!(index_r(&v1, 2), &Word8::of(2));
/// }
/// ```
pub fn index_r<N, const C: usize>(v: &VectorR<N, C>, column_index: usize) -> &N {
    &v[(1, column_index)]
}

/// Used to obtain the basis vector.
///
/// The `index` represents the index of the basis vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{basis_vector, VectorC},
///     number::instances::int8::Int8,
/// };
///
/// pub fn main() {
///     let e1_a: VectorC<Int8, 3> = VectorC::of(&[Int8::of(0), Int8::of(1), Int8::of(0)]).unwrap();
///     let e1_b = basis_vector::<Int8, 3>(2);
///     assert_eq!(e1_a, e1_b);
/// }
/// ```
pub fn basis_vector<N, const R: usize>(index: usize) -> VectorC<N, R>
where
    N: Number,
{
    let mut basis = VectorC::<N, R>::default();
    basis[(index, 1)] = N::one();
    basis
}

/// Calculate the dot product of two column vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{dot_product, VectorC},
///     number::instances::int8::Int8,
/// };
///
/// fn main() {
///     let v1: VectorC<Int8, 3> = VectorC::of(&[Int8::of(1), Int8::of(2), Int8::of(3)]).unwrap();
///     let v2: VectorC<Int8, 3> = VectorC::of(&[Int8::of(4), Int8::of(5), Int8::of(6)]).unwrap();
///     assert_eq!(dot_product(&v1, &v2), Int8::of(32));
/// }
/// ```
pub fn dot_product<N, const R: usize>(v1: &VectorC<N, R>, v2: &VectorC<N, R>) -> N
where
    N: Real,
{
    v1.inner
        .par_iter()
        .zip(v2.inner.par_iter())
        .map(|(e1, e2)| e1.clone() * e2.clone())
        .reduce(|| N::zero(), |acc, e| acc + e)
}

/// Calculate the cross product of two three-dimensional column vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{cross_product, VectorC},
///     number::instances::int8::Int8,
/// };
///
/// fn main() {
///     let v1: VectorC<Int8, 3> = VectorC::of(&[Int8::of(1), Int8::of(2), Int8::of(3)]).unwrap();
///     let v2: VectorC<Int8, 3> = VectorC::of(&[Int8::of(4), Int8::of(5), Int8::of(6)]).unwrap();
///     assert_eq!(
///         cross_product(v1, v2),
///         VectorC::<Int8, 3>::of(&[Int8::of(-3), Int8::of(6), Int8::of(-3)]).unwrap()
///     );
/// }
/// ```
pub fn cross_product<N>(v1: VectorC<N, 3>, v2: VectorC<N, 3>) -> VectorC<N, 3>
where
    N: Number,
{
    let mut inner = vec![N::zero(); 3];
    inner[0] = index_c(&v1, 2).clone() * index_c(&v2, 3).clone()
        - index_c(&v1, 3).clone() * index_c(&v2, 2).clone();
    inner[1] = index_c(&v1, 3).clone() * index_c(&v2, 1).clone()
        - index_c(&v1, 1).clone() * index_c(&v2, 3).clone();
    inner[2] = index_c(&v1, 1).clone() * index_c(&v2, 2).clone()
        - index_c(&v1, 2).clone() * index_c(&v2, 1).clone();

    VectorC { inner }
}

/// Construct a matrix using one column vector and one row vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         matrix::Matrix,
///         vector::{layer_product, VectorC, VectorR},
///     },
///     number::instances::int8::Int8,
/// };
///
/// fn main() {
///     let v1: VectorC<Int8, 3> = VectorC::of(&[Int8::of(1), Int8::of(2), Int8::of(3)]).unwrap();
///     let v2: VectorR<Int8, 3> = VectorR::of(&[Int8::of(4), Int8::of(5), Int8::of(6)]).unwrap();
///     let data = [
///         4i8, 5i8, 6i8, // row1
///         8i8, 10i8, 12i8, // row2
///         12i8, 15i8, 18i8, // row3
///     ]
///     .map(|e| Int8::of(e));
///     let m = Matrix::<Int8, 3, 3>::of(&data).unwrap();
///     assert_eq!(layer_product(&v1, &v2), m);
/// }
/// ```
pub fn layer_product<N, const R: usize, const C: usize>(
    v1: &VectorC<N, R>,
    v2: &VectorR<N, C>,
) -> Matrix<N, R, C>
where
    N: Number,
{
    let mut inner = Vec::with_capacity(R * C);
    for row_index in 1..=R {
        for column_index in 1..=C {
            inner.push(index_c(&v1, row_index).clone() * index_r(&v2, column_index).clone());
        }
    }
    Matrix { inner }
}

/// Calculate the convolution of two column vectors.
///
/// # Panics
///
/// This function requires the use of the `#![feature(generic_const_exprs)]`.
///
/// # Examples
///
/// ```rust
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use rmatrix_ks::{
///     matrix::vector::{convolution, VectorC},
///     number::instances::int8::Int8,
/// };
///
/// fn main() {
///     let v1: VectorC<Int8, 3> = VectorC::of(&[Int8::of(1), Int8::of(2), Int8::of(3)]).unwrap();
///     let v2: VectorC<Int8, 2> = VectorC::of(&[Int8::of(4), Int8::of(5)]).unwrap();
///     let cv = VectorC::<Int8, 4>::of(&[4, 13, 22, 15].map(|e| Int8::of(e))).unwrap();
///     assert_eq!(convolution(v1, v2), cv);
/// }
/// ```
pub fn convolution<N, const R1: usize, const R2: usize>(
    v1: VectorC<N, R1>,
    v2: VectorC<N, R2>,
) -> VectorC<N, { R1 + R2 - 1 }>
where
    N: Number,
{
    let mut inner = Vec::with_capacity(R1 + R2 - 1);
    for index in 1..=(R1 + R2 - 1) {
        let mut sum = N::zero();
        for p in 1..=R1.min(index) {
            let q = index + 1 - p;
            if (1..=R2).contains(&q) {
                sum = sum + index_c(&v1, p).clone() * index_c(&v2, q).clone();
            }
        }
        inner.push(sum);
    }
    VectorC { inner }
}

/// Calculate the Euclidean norm.
///
/// Aka L2-norm.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{euclidean_norm, VectorC},
///     number::{
///         instances::float::Float,
///         traits::{floating::Floating, zero::Zero},
///     },
/// };
///
/// fn main() {
///     let v: VectorC<Float, 3> =
///         VectorC::of(&[Float::of(1.0), Float::of(2.0), Float::of(3.0)]).unwrap();
///     assert!((euclidean_norm(&v) - Float::of(14.0).square_root()).is_zero());
/// }
/// ```
pub fn euclidean_norm<N, const R: usize>(v: &VectorC<N, R>) -> N
where
    N: RealFloat,
{
    v.inner
        .par_iter()
        .map(|e| e.clone() * e.clone())
        .reduce(|| N::zero(), |acc, e| acc + e)
        .square_root()
}

/// Calculate the maximum norm.
///
/// Aka L_inf-norm.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{maximum_norm, VectorC},
///     number::instances::int8::Int8,
/// };
///
/// fn main() {
///     let v = VectorC::<Int8, 3>::of(&[Int8::of(2), Int8::of(-5), Int8::of(3)]).unwrap();
///     assert_eq!(maximum_norm(&v), Int8::of(5))
/// }
/// ```
pub fn maximum_norm<N, const R: usize>(v: &VectorC<N, R>) -> N
where
    N: Real,
{
    let mut norm = N::zero();
    for e in v.linear_iter().map(|e| e.absolute_value()) {
        if e > norm {
            norm = e;
        }
    }
    norm
}

/// Calculate the root mean square of the column vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{root_mean_square, VectorC},
///     number::{
///         instances::float::Float,
///         traits::{floating::Floating, zero::Zero},
///     },
/// };
///
/// fn main() {
///     let v: VectorC<Float, 3> =
///         VectorC::of(&[Float::of(1.0), Float::of(2.0), Float::of(3.0)]).unwrap();
///     assert!((root_mean_square(&v) - Float::of(14.0 / 3.0).square_root()).is_zero());
/// }
/// ```
pub fn root_mean_square<N, const R: usize>(v: &VectorC<N, R>) -> N
where
    N: RealFloat,
{
    let l2_norm = euclidean_norm(v);
    let r_sqrt = N::from_integer(Integer::of_str(&format!("{}", R)).expect(&format!(
        "Error[matrix::vector::root_mean_square]: Failed to convert {} from usize to Integer.",
        R
    )))
    .square_root();
    l2_norm / r_sqrt
}

/// Calculate the angle between two vectors.
///
/// Expressed in radians.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{angle_between, VectorC},
///     number::{instances::double::Double, traits::zero::Zero},
/// };
///
/// fn main() {
///     let v1: VectorC<Double, 3> =
///         VectorC::of(&[Double::of(1.0), Double::of(5.0), Double::of(4.0)]).unwrap();
///     let v2: VectorC<Double, 3> =
///         VectorC::of(&[Double::of(8.0), Double::of(-4.0), Double::of(3.0)]).unwrap();
///     let angle = angle_between(&v1, &v2);
///     assert!(angle.is_some());
///     assert!((angle.unwrap() - Double::of(core::f64::consts::PI / 2.0)).is_zero());
///
///     // angle between a vector and itself is zero
///     let self_angle = angle_between(&v1, &v1);
///     assert!(self_angle.is_some());
///     assert!(self_angle.unwrap().is_zero());
///
///     // zero-vector has no angle with any other vector
///     let v3 = VectorC::of(&[Double::zero(), Double::zero(), Double::zero()]).unwrap();
///     assert!(angle_between(&v1, &v3).is_none());
/// }
/// ```
pub fn angle_between<N, const R: usize>(v1: &VectorC<N, R>, v2: &VectorC<N, R>) -> Option<N>
where
    N: RealFloat,
{
    let v1_norm = euclidean_norm(v1);
    let v2_norm = euclidean_norm(v2);
    if v1_norm.is_zero() || v2_norm.is_zero() {
        eprintln!(concat!(
            "Error[matrix::vector::angle_between]: ",
            "The zero-vector has no angle with other vectors."
        ));
        None
    } else {
        Some((dot_product(v1, v2) / (v1_norm * v2_norm)).arc_cosine())
    }
}

/// Project one vector onto another vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{project_to, VectorC},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let v1 = VectorC::<Float, 2>::of(&[3.0, 4.0].map(Float::of)).unwrap();
///     let v2 = VectorC::<Float, 2>::of(&[1.0, 2.0].map(Float::of)).unwrap();
///     let p = project_to(&v1, &v2);
///     assert_eq!(
///         p,
///         VectorC::<Float, 2>::of(&[2.2, 4.4].map(Float::of)).unwrap()
///     );
/// }
/// ```
pub fn project_to<N, const R: usize>(from: &VectorC<N, R>, to: &VectorC<N, R>) -> VectorC<N, R>
where
    N: RealFrac,
{
    let p1 = dot_product(to, from);
    let p2 = dot_product(to, to);
    to.clone() * (p1 / p2)
}
