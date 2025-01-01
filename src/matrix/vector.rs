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

/// Constructing a row vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, vector::row_vector},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let rv = row_vector(3, &[1, 2, 3].map(Word8::of));
///     let rv_expect = Matrix::of(1, 3, &[1, 2, 3].map(Word8::of));
///     assert_eq!(rv, rv_expect);
/// }
/// ```
pub fn row_vector<N>(dim: usize, data: &[N]) -> Option<Matrix<N>>
where
    N: Clone,
{
    Matrix::of(1, dim, data)
}
/// Determine whether the given matrix is a row vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         matrix::Matrix,
///         vector::{is_row_vector, row_vector},
///     },
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let v = row_vector(3, &[1, 0, 0].map(Word8::of)).unwrap();
///     assert!(is_row_vector(&v));
///     let m = Matrix::of(2, 2, &[1, 0, 2, 5].map(Word8::of)).unwrap();
///     assert!(!is_row_vector(&m));
/// }
/// ```
pub fn is_row_vector<N>(v: &Matrix<N>) -> bool { v.row == 1 }

/// Constructing a column vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, vector::column_vector},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let cv = column_vector(3, &[1, 2, 3].map(Word8::of));
///     let cv_expect = Matrix::of(3, 1, &[1, 2, 3].map(Word8::of));
///     assert_eq!(cv, cv_expect);
/// }
/// ```
pub fn column_vector<N>(dim: usize, data: &[N]) -> Option<Matrix<N>>
where
    N: Clone,
{
    Matrix::of(dim, 1, data)
}

/// Determine whether the given matrix is a column vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         matrix::Matrix,
///         vector::{column_vector, is_column_vector},
///     },
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let v = column_vector(3, &[1, 0, 0].map(Word8::of)).unwrap();
///     assert!(is_column_vector(&v));
///     let m = Matrix::of(2, 2, &[1, 0, 2, 5].map(Word8::of)).unwrap();
///     assert!(!is_column_vector(&m));
/// }
/// ```
pub fn is_column_vector<N>(v: &Matrix<N>) -> bool { v.column == 1 }

/// Used to obtain the basis vector.
///
/// The `index` represents the index of the basis vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, vector::basis_vector},
///     number::instances::int8::Int8,
/// };
///
/// pub fn main() {
///     let e1_a: Matrix<Int8> =
///         Matrix::of(3, 1, &[Int8::of(0), Int8::of(1), Int8::of(0)]).unwrap();
///     let e1_b = basis_vector::<Int8>(3, 2);
///     assert_eq!(e1_a, e1_b);
/// }
/// ```
pub fn basis_vector<N>(dim: usize, index: usize) -> Matrix<N>
where
    N: Number,
{
    let mut basis = Matrix::<N>::defaults(dim, 1);
    basis[(index, 1)] = N::one();
    basis
}

/// Calculate the dot product of two column vectors.
///
/// # Panics
///
/// The dimensions of the column vectors involved in the function must match.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{column_vector, dot_product},
///     number::instances::int8::Int8,
/// };
///
/// fn main() {
///     let v1 = column_vector::<Int8>(3, &[Int8::of(1), Int8::of(2), Int8::of(3)]).unwrap();
///     let v2 = column_vector::<Int8>(3, &[Int8::of(4), Int8::of(5), Int8::of(6)]).unwrap();
///     assert_eq!(dot_product(&v1, &v2), Int8::of(32));
/// }
/// ```
pub fn dot_product<N>(v1: &Matrix<N>, v2: &Matrix<N>) -> N
where
    N: Real,
{
    assert!(
        is_column_vector(v1) && is_column_vector(v2) && v1.row == v2.row,
        concat!(
            "Error[matrix::vector::dot_product]: ",
            "Only column vectors with matching dimensions can be multiplied."
        )
    );

    v1.inner
        .par_iter()
        .zip(v2.inner.par_iter())
        .map(|(e1, e2)| e1.clone() * e2.clone())
        .reduce(|| N::zero(), |acc, e| acc + e)
}

/// Calculate the cross product of two 3-dim column vectors.
///
/// # Panics
///
/// Only 3-dim column vectors can be crossed.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{column_vector, cross_product},
///     number::instances::int8::Int8,
/// };
///
/// fn main() {
///     let v1 = column_vector::<Int8>(3, &[Int8::of(1), Int8::of(2), Int8::of(3)]).unwrap();
///     let v2 = column_vector::<Int8>(3, &[Int8::of(4), Int8::of(5), Int8::of(6)]).unwrap();
///     assert_eq!(
///         cross_product(v1, v2),
///         column_vector::<Int8>(3, &[Int8::of(-3), Int8::of(6), Int8::of(-3)]).unwrap()
///     );
/// }
/// ```
pub fn cross_product<N>(v1: Matrix<N>, v2: Matrix<N>) -> Matrix<N>
where
    N: Number,
{
    assert!(
        v1.shape() == (3, 1) && v2.shape() == (3, 1),
        concat!(
            "Error[matrix::vector::cross_product]: ",
            "Only three-dimensional column vectors can be crossed."
        )
    );

    let mut inner = vec![N::zero(); 3];
    inner[0] =
        v1[(2, 1)].clone() * v2[(3, 1)].clone() - v1[(3, 1)].clone() * v2[(2, 1)].clone();
    inner[1] =
        v1[(3, 1)].clone() * v2[(1, 1)].clone() - v1[(1, 1)].clone() * v2[(3, 1)].clone();
    inner[2] =
        v1[(1, 1)].clone() * v2[(2, 1)].clone() - v1[(2, 1)].clone() * v2[(1, 1)].clone();
    Matrix {
        inner,
        row: 3,
        column: 1,
    }
}

/// Construct a matrix using one column vector and one row vector.
///
/// # Panics
///
/// Only one column vector and one row vector can be used for the layer product.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         matrix::Matrix,
///         vector::{column_vector, layer_product, row_vector},
///     },
///     number::instances::int8::Int8,
/// };
///
/// fn main() {
///     let v1 = column_vector::<Int8>(3, &[Int8::of(1), Int8::of(2), Int8::of(3)]).unwrap();
///     let v2 = row_vector::<Int8>(3, &[Int8::of(4), Int8::of(5), Int8::of(6)]).unwrap();
///     let data = [
///         4i8, 5i8, 6i8, // row1
///         8i8, 10i8, 12i8, // row2
///         12i8, 15i8, 18i8, // row3
///     ]
///     .map(|e| Int8::of(e));
///     let m = Matrix::<Int8>::of(3, 3, &data).unwrap();
///     assert_eq!(layer_product(&v1, &v2), m);
/// }
/// ```
pub fn layer_product<N>(v1: &Matrix<N>, v2: &Matrix<N>) -> Matrix<N>
where
    N: Number,
{
    assert!(
        is_column_vector(v1) && is_row_vector(v2),
        concat!(
            "Error[matrix::vector::layer_product]: ",
            "Only one column vector and one row vector ",
            "can be used for the layer product."
        )
    );

    let mut inner = Vec::with_capacity(v1.row * v2.column);
    for row_index in 1..=v1.row {
        for column_index in 1..=v2.column {
            inner.push(v1[(row_index, 1)].clone() * v2[(1, column_index)].clone());
        }
    }
    Matrix {
        inner,
        row: v1.row,
        column: v2.column,
    }
}

/// Calculate the convolution of two column vectors.
///
/// # Panics
///
/// Only column vectors can be convolved.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{column_vector, convolution},
///     number::instances::int8::Int8,
/// };
///
/// fn main() {
///     let v1 = column_vector::<Int8>(3, &[Int8::of(1), Int8::of(2), Int8::of(3)]).unwrap();
///     let v2 = column_vector::<Int8>(2, &[Int8::of(4), Int8::of(5)]).unwrap();
///     let cv = column_vector::<Int8>(4, &[4, 13, 22, 15].map(|e| Int8::of(e))).unwrap();
///     assert_eq!(convolution(&v1, &v2), cv);
/// }
/// ```
pub fn convolution<N>(v1: &Matrix<N>, v2: &Matrix<N>) -> Matrix<N>
where
    N: Number,
{
    assert!(
        is_column_vector(v1) && is_column_vector(v2),
        concat!(
            "Error[matrix::vector::convolution]: ",
            "Only column vectors can be convolved."
        )
    );

    let mut inner = Vec::with_capacity(v1.row + v2.row - 1);
    for index in 1..=(v1.row + v2.row - 1) {
        let mut sum = N::zero();
        for p in 1..=v1.row.min(index) {
            let q = index + 1 - p;
            if (1..=v2.row).contains(&q) {
                sum = sum + v1[(p, 1)].clone() * v2[(q, 1)].clone();
            }
        }
        inner.push(sum);
    }
    Matrix {
        inner,
        row: v1.row + v2.row - 1,
        column: 1,
    }
}

/// Calculate the Euclidean norm.
///
/// Aka L2-norm.
///
/// # Panics
///
/// Only vectors have the Euclidean norm.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{column_vector, euclidean_norm},
///     number::{
///         instances::float::Float,
///         traits::{floating::Floating, zero::Zero},
///     },
/// };
///
/// fn main() {
///     let v = column_vector::<Float>(3, &[Float::of(1.0), Float::of(2.0), Float::of(3.0)])
///         .unwrap();
///     assert!((euclidean_norm(&v) - Float::of(14.0).square_root()).is_zero());
/// }
/// ```
pub fn euclidean_norm<N>(v: &Matrix<N>) -> N
where
    N: RealFloat,
{
    assert!(
        is_row_vector(v) || is_column_vector(v),
        concat!(
            "Error[matrix::vector::euclidean_norm]: ",
            "Only vectors have the Euclidean norm."
        )
    );

    v.inner
        .par_iter()
        .map(|e| e.clone() * e.clone())
        .reduce(|| N::zero(), |acc, e| acc + e)
        .square_root()
}

/// Normalize the real vector.
///
/// # Panics
///
/// This function can only be used for column vector normalization.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{column_vector, normalize},
///     number::{instances::float::Float, traits::floating::Floating},
/// };
///
/// fn main() {
///     let v1 = column_vector::<Float>(3, &[1.0, 1.0, 1.0].map(Float::of)).unwrap();
///     let normalized = normalize(&v1);
///     let normalized_expect = v1 / Float::of(3.0).square_root();
///     assert_eq!(normalized, normalized_expect);
/// }
/// ```
pub fn normalize<N>(v: &Matrix<N>) -> Matrix<N>
where
    N: RealFloat,
{
    assert!(
        is_column_vector(v),
        concat!(
            "Error[matrix::vector::normalize]: ",
            "This function can only be used for column vector normalization."
        )
    );

    if v.inner.par_iter().all(|e| e.is_zero()) {
        Matrix::defaults(v.row, 1)
    } else {
        let v_norm = euclidean_norm(v);
        v.clone() / v_norm
    }
}

/// Calculate the maximum norm.
///
/// Aka L_inf-norm.
///
/// # Panics
///
/// This function can only be used for column vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{column_vector, maximum_norm},
///     number::instances::int8::Int8,
/// };
///
/// fn main() {
///     let v = column_vector::<Int8>(3, &[Int8::of(2), Int8::of(-5), Int8::of(3)]).unwrap();
///     assert_eq!(maximum_norm(&v), Int8::of(5))
/// }
/// ```
pub fn maximum_norm<N>(v: &Matrix<N>) -> N
where
    N: Real,
{
    assert!(
        is_column_vector(v),
        concat!(
            "Error[matrix::vector::maximum_norm]: ",
            "This function can only be used for column vectors."
        )
    );

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
/// # Panics
///
/// This function can only be used for column vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{column_vector, root_mean_square},
///     number::{
///         instances::float::Float,
///         traits::{floating::Floating, zero::Zero},
///     },
/// };
///
/// fn main() {
///     let v = column_vector::<Float>(3, &[Float::of(1.0), Float::of(2.0), Float::of(3.0)])
///         .unwrap();
///     assert!((root_mean_square(&v) - Float::of(14.0 / 3.0).square_root()).is_zero());
/// }
/// ```
pub fn root_mean_square<N>(v: &Matrix<N>) -> N
where
    N: RealFloat,
{
    assert!(
        is_column_vector(v),
        concat!(
            "Error[matrix::vector::root_mean_square]: ",
            "This function can only be used for column vectors."
        )
    );

    let l2_norm = euclidean_norm(v);
    let r_sqrt = N::from_integer(Integer::of_str(&format!("{}", v.row)).expect(&format!(
        "Error[matrix::vector::root_mean_square]: Failed to convert {} from usize to Integer.",
        v.row
    )))
    .square_root();
    l2_norm / r_sqrt
}

/// Calculate the angle between two vectors.
///
/// Expressed in radians.
///
/// # Panics
///
/// This function can only be used for column vectors with matching dimensions.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{angle_between, column_vector},
///     number::{instances::double::Double, traits::zero::Zero},
/// };
///
/// fn main() {
///     let v1 =
///         column_vector::<Double>(3, &[Double::of(1.0), Double::of(5.0), Double::of(4.0)])
///             .unwrap();
///     let v2 =
///         column_vector::<Double>(3, &[Double::of(8.0), Double::of(-4.0), Double::of(3.0)])
///             .unwrap();
///     let angle = angle_between(&v1, &v2);
///     assert!(angle.is_some());
///     assert_eq!(angle.unwrap(), Double::of(core::f64::consts::PI / 2.0));
///
///     // angle between a vector and itself is zero
///     let self_angle = angle_between(&v1, &v1);
///     assert!(self_angle.is_some());
///     assert!(self_angle.unwrap().is_zero());
///
///     // zero-vector has no angle with any other vector
///     let v3 = column_vector::<Double>(3, &[Double::zero(), Double::zero(), Double::zero()])
///         .unwrap();
///     assert!(angle_between(&v1, &v3).is_none());
/// }
/// ```
pub fn angle_between<N>(v1: &Matrix<N>, v2: &Matrix<N>) -> Option<N>
where
    N: RealFloat,
{
    assert!(
        is_column_vector(v1) && is_column_vector(v2) && v1.row == v2.row,
        concat!(
            "Error[matrix::vector::angle_between]: ",
            "This function can only be used for column vectors with matching dimensions."
        )
    );

    let zero_vector = Matrix::defaults(v1.row, 1);
    if v1 == &zero_vector || v2 == &zero_vector {
        eprintln!(concat!(
            "Error[matrix::vector::angle_between]: ",
            "The zero-vector has no angle with other vectors."
        ));
        None
    } else if v1 == v2 {
        Some(N::zero())
    } else {
        Some((dot_product(&normalize(v1), &normalize(v2))).arc_cosine())
    }
}

/// Project one vector onto another vector.
///
/// # Panics
///
/// This function can only be used for column vectors with matching dimensions.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::vector::{column_vector, project_to},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let v1 = column_vector::<Float>(2, &[3.0, 4.0].map(Float::of)).unwrap();
///     let v2 = column_vector::<Float>(2, &[1.0, 2.0].map(Float::of)).unwrap();
///     let p = project_to(&v1, &v2);
///     assert_eq!(
///         p,
///         column_vector::<Float>(2, &[2.2, 4.4].map(Float::of)).unwrap()
///     );
/// }
/// ```
pub fn project_to<N>(from: &Matrix<N>, to: &Matrix<N>) -> Matrix<N>
where
    N: RealFrac,
{
    assert!(
        is_column_vector(from) && is_column_vector(to) && from.row == to.row,
        concat!(
            "Error[matrix::vector::project_to]: ",
            "This function can only be used for column vectors with matching dimensions."
        )
    );

    let p1 = dot_product(to, from);
    let p2 = dot_product(to, to);
    to.clone() * (p1 / p2)
}
