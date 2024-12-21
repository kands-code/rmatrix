//! # matrix::matrix
//!
//! Definition of the matrix along with related functions and implementations.

use crate::{
    matrix::{
        utils::apply,
        vector::{layer_product, VectorC, VectorR},
    },
    number::traits::{fractional::Fractional, number::Number},
};

use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

/// A matrix is a container of a single type that has two dimensions: rows and columns.
#[derive(Clone)]
pub struct Matrix<N, const R: usize, const C: usize> {
    /// Internal container.
    pub(crate) inner: Vec<N>,
}

impl<N, const R: usize, const C: usize> Matrix<N, R, C> {
    /// Convert the matrix element positions to internal container indices.
    pub(crate) fn position_to_index(row_index: usize, column_index: usize) -> usize {
        (row_index - 1) * C + column_index - 1
    }

    /// Get the length of the matrix diagonal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     assert_eq!(Matrix::<Word8, 3, 2>::get_diagonal_length(), 2);
    /// }
    /// ```
    pub const fn get_diagonal_length() -> usize {
        if R > C {
            C
        } else {
            R
        }
    }

    /// Construct the matrix by passing in data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8, 2, 2>::of(&[1, 2, 3, 4, 5, 6].map(|e| Word8::of(e))).unwrap();
    ///     // Data will be truncated when the length exceeds the number of matrix elements.
    ///     assert_eq!(
    ///         m.linear_iter().cloned().collect::<Vec<Word8>>(),
    ///         [1, 2, 3, 4].map(|e| Word8::of(e)).to_vec()
    ///     );
    ///     let n = Matrix::<Word8, 2, 2>::of(&[1, 2, 3].map(|e| Word8::of(e)));
    ///     // Returns None when the data length is less than the number of matrix elements.
    ///     assert_eq!(n, None);
    /// }
    /// ```
    pub fn of(data: &[N]) -> Option<Self>
    where
        N: Clone,
    {
        if data.len() < R * C {
            eprintln!(
                concat!(
                    "Error[Matrix::of]: ",
                    "The data length ({}) does not meet the required number ",
                    "of elements ({}) for the matrix."
                ),
                data.len(),
                R * C
            );
            None
        } else {
            Some(Self {
                inner: Vec::from(&data[..R * C]),
            })
        }
    }

    /// Returns a matrix with ONEs on the main diagonalcorresponding to the specified shape and ZEROs elsewhere,
    /// essentially a type of identity matrix.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let i = Matrix::<Word8, 2, 2>::eyes();
    ///     let m = Matrix::<Word8, 2, 2>::of(&[1, 0, 0, 1].map(|e| Word8::of(e))).unwrap();
    ///     assert_eq!(i, m);
    /// }
    /// ```
    pub fn eyes() -> Self
    where
        N: Number,
    {
        let mut id_mat = Self::default();
        for index in 1..=Self::get_diagonal_length() {
            id_mat[(index, index)] = N::one();
        }
        id_mat
    }

    /// Uses the passed data as the main diagonal, with other elements being ZEROs,
    /// resulting in a type of diagonal matrix.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8, 2, 3>::diagonal(&[1, 2].map(|e| Word8::of(e))).unwrap();
    ///     let d = Matrix::<Word8, 2, 3>::of(&[1, 0, 0, 0, 2, 0].map(|e| Word8::of(e))).unwrap();
    ///     assert_eq!(m, d);
    ///     // Returns None when the data length is less than the length of the matrix diagonal.
    ///     let n = Matrix::<Word8, 2, 2>::of(&[1].map(|e| Word8::of(e)));
    ///     assert_eq!(n, None);
    /// }
    /// ```
    pub fn diagonal(data: &[N]) -> Option<Self>
    where
        N: Clone + Default,
    {
        let length = Self::get_diagonal_length();
        if data.len() < length {
            eprintln!(
                concat!(
                    "Error[Matrix::diagonal]: ",
                    "The data length ({}) does not meet the required number ",
                    "of elements ({}) for the matrix."
                ),
                data.len(),
                length
            );
            None
        } else {
            let mut diag = Self::default();
            for index in 1..=Self::get_diagonal_length() {
                diag[(index, index)] = data[index - 1].clone();
            }
            Some(diag)
        }
    }

    /// Constructs a permutation matrix for row exchanges.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
    ///     let p = Matrix::<Word8, 3, 3>::p_change(1, 3);
    ///     let n = Matrix::<Word8, 3, 3>::of(&[7, 8, 9, 4, 5, 6, 1, 2, 3].map(|e| Word8::of(e))).unwrap();
    ///     assert_eq!(p * m, n);
    /// }
    /// ```
    pub fn p_change(row1: usize, row2: usize) -> Matrix<N, R, R>
    where
        N: Number,
    {
        let mut p_mat = Matrix::eyes();
        p_mat[(row1, row1)] = N::zero();
        p_mat[(row2, row2)] = N::zero();
        p_mat[(row1, row2)] = N::one();
        p_mat[(row2, row1)] = N::one();
        p_mat
    }

    /// Constructs a permutation matrix that multiplies the elements of a specific row by a given value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
    ///     let p = Matrix::<Word8, 3, 3>::p_muls(2, Word8::of(2));
    ///     let n =
    ///         Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 8, 10, 12, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
    ///     assert_eq!(p * m, n);
    /// }
    /// ```
    pub fn p_muls(row: usize, scalar: N) -> Matrix<N, R, R>
    where
        N: Number,
    {
        let mut p_mat = Matrix::eyes();
        p_mat[(row, row)] = scalar;
        p_mat
    }

    /// Constructs a permutation matrix that adds the elements of a specific row,
    /// multiplied by a given value, to the elements of another row.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::int8::Int8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Int8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Int8::of(e))).unwrap();
    ///     let p = Matrix::<Int8, 3, 3>::p_add(1, 2, Int8::of(-4));
    ///     let n = Matrix::<Int8, 3, 3>::of(&[1, 2, 3, 0, -3, -6, 7, 8, 9].map(|e| Int8::of(e))).unwrap();
    ///     assert_eq!(p * m, n);
    /// }
    /// ```
    pub fn p_add(from: usize, to: usize, scalar: N) -> Matrix<N, R, R>
    where
        N: Number,
    {
        let mut p_mat = Matrix::eyes();
        p_mat[(to, from)] = scalar;
        p_mat
    }

    /// Constructs a matrix of a specific shape, where each element is randomly selected from a given range.
    ///
    /// The `rand` library is used, and the matrix element type must implement the `SampleUniform` trait.
    /// In particular, the `Ratio` type cannot be used.
    ///
    /// For the `Complex` type, the ranges for the elements are defined by the real and imaginary parts.
    /// For example, if `lb` is passed as `Complex(-1.0, -3.0)` and `ub` as `Complex(3.0, 1.0)`,
    /// the real part of the matrix elements will be randomly selected from the range `[-1.0, 3.0)`,
    /// while the imaginary part will be randomly selected from the range `[-3.0, 1.0)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{
    ///     matrix::matrix::Matrix,
    ///     number::instances::{complex::Complex, float::Float},
    /// };
    ///
    /// fn main() {
    ///     let m = Matrix::<Complex<Float>, 3, 3>::rand(
    ///         Complex::of(Float::of(-1.0), Float::of(-3.0)),
    ///         Complex::of(Float::of(3.0), Float::of(1.0)),
    ///     );
    ///     assert!(
    ///         m.linear_iter().all(|e| Float::of(-1.0) < e.real
    ///         && e.real < Float::of(3.0) // real range
    ///         && Float::of(-3.0) < e.imaginary
    ///         && e.imaginary < Float::of(1.0)) // imaginary range
    ///     );
    /// }
    /// ```
    pub fn rand(lb: N, ub: N) -> Self
    where
        N: Number + SampleUniform,
    {
        let range = Uniform::new(lb, ub);
        let mut rng = rand::thread_rng();
        let inner = range.sample_iter(&mut rng).take(R * C).collect();
        Self { inner }
    }

    /// Returns the shape of the matrix, specifically the number of rows and columns.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8, 2, 3>::of(&[1, 2, 3, 4, 5, 6].map(|e| Word8::of(e))).unwrap();
    ///     assert_eq!(m.shape(), (2, 3));
    /// }
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (R, C)
    }

    /// Returns the internal data of the matrix as an iterator.
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let data = [1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e));
    ///     let m = Matrix::<Word8, 3, 3>::of(&data).unwrap();
    ///     assert!(m.linear_iter().zip(data.iter()).all(|(e1, e2)| e1 == e2));
    /// }
    /// ```
    pub fn linear_iter(&self) -> std::slice::Iter<'_, N> {
        self.inner.iter()
    }

    /// Retrieves the element at a specific position in the matrix.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
    ///     assert_eq!(m.get(2, 1), Some(&Word8::of(4)));
    ///     assert_eq!(m.get(3, 2), Some(&Word8::of(8)));
    ///     // Returns `None` when the position exceeds the boundaries.
    ///     assert_eq!(m.get(4, 2), None);
    /// }
    /// ```
    pub fn get(&self, row_index: usize, column_index: usize) -> Option<&N> {
        if row_index == 0 || column_index == 0 || row_index > R || column_index > C {
            eprintln!(
                "Error[Matrix::get]: Index ({}, {}) is out of bounds.",
                row_index, column_index
            );
            None
        } else {
            let position = Self::position_to_index(row_index, column_index);
            self.linear_iter().nth(position)
        }
    }

    /// Sets the value of the element at a specific position in the matrix.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let mut m =
    ///         Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
    ///     m.set(2, 1, Word8::of(12));
    ///     m.set(3, 2, Word8::of(16));
    ///     let n =
    ///         Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 12, 5, 6, 7, 16, 9].map(|e| Word8::of(e))).unwrap();
    ///     assert_eq!(m, n);
    /// }
    /// ```
    pub fn set(&mut self, row_index: usize, column_index: usize, value: N) {
        if row_index == 0 || column_index == 0 || row_index > R || column_index > C {
            eprintln!(
                "Error[Matrix::set]: Index ({}, {}) is out of bounds.",
                row_index, column_index
            );
        } else {
            let position = Self::position_to_index(row_index, column_index);
            self.inner[position] = value;
        }
    }

    /// Retrieves a specific row of the matrix and returns it as a row vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{
    ///     matrix::{matrix::Matrix, vector::VectorR},
    ///     number::instances::word8::Word8,
    /// };
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
    ///     // What is retrieved is a reference to the element, not the value of the element.
    ///     let r2 = m.get_row(2).unwrap();
    ///     let v1 = Word8::of(4);
    ///     let v2 = Word8::of(5);
    ///     let v3 = Word8::of(6);
    ///     let r2_expect = VectorR::<&Word8, 3>::of(&[&v1, &v2, &v3]).unwrap();
    ///     assert_eq!(r2, r2_expect);
    ///     // Returns `None` when the position exceeds the boundaries.
    ///     let n = m.get_row(4);
    ///     assert_eq!(n, None);
    /// }
    /// ```
    pub fn get_row(&self, row_index: usize) -> Option<VectorR<&N, C>> {
        if row_index == 0 || row_index > R {
            eprintln!(
                "Error[Matrix::get_row]: Index ({}) is out of bounds.",
                row_index
            );
            None
        } else {
            let mut inner = Vec::with_capacity(C);
            for column_index in 1..=C {
                inner.push(&self[(row_index, column_index)]);
            }
            Some(VectorR { inner })
        }
    }

    /// Retrieves a specific column of the matrix and returns it as a column vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{
    ///     matrix::{matrix::Matrix, vector::VectorC},
    ///     number::instances::word8::Word8,
    /// };
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
    ///     // What is retrieved is a reference to the element, not the value of the element.
    ///     let c2 = m.get_column(2).unwrap();
    ///     let v1 = Word8::of(2);
    ///     let v2 = Word8::of(5);
    ///     let v3 = Word8::of(8);
    ///     let c2_expect = VectorC::<&Word8, 3>::of(&[&v1, &v2, &v3]).unwrap();
    ///     assert_eq!(c2, c2_expect);
    ///     // Returns `None` when the position exceeds the boundaries.
    ///     let n = m.get_column(4);
    ///     assert_eq!(n, None);
    /// }
    /// ```
    pub fn get_column(&self, column_index: usize) -> Option<VectorC<&N, R>> {
        if column_index == 0 || column_index > C {
            eprintln!(
                "Error[Matrix::get_column]: Index ({}) is out of bounds.",
                column_index
            );
            None
        } else {
            let mut inner = Vec::with_capacity(R);
            for row_index in 1..=R {
                inner.push(&self[(row_index, column_index)]);
            }
            Some(VectorC { inner })
        }
    }

    /// Determines whether two matrices are equal.
    ///
    /// For elements of the Floating type, which cannot be directly compared for equality using `==`,
    /// this function can be used to assess their equality.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{
    ///     matrix::matrix::Matrix,
    ///     number::{instances::float::Float, traits::one::One},
    /// };
    ///
    /// fn main() {
    ///     let m = Matrix::<Float, 2, 2>::of(&[1.0, 2.0, 4.0, 6.0].map(|e| Float::of(e))).unwrap();
    ///     let p = Matrix::<Float, 2, 2>::p_add(1, 2, Float::one());
    ///     let n = Matrix::<Float, 2, 2>::of(&[1.0, 2.0, 3.0, 4.0].map(|e| Float::of(e))).unwrap();
    ///     assert!(m.equals(&(p * n)));
    /// }
    /// ```
    pub fn equals(&self, rhs: &Self) -> bool
    where
        N: Number,
    {
        self.inner
            .par_iter()
            .zip(rhs.inner.par_iter())
            .all(|(e1, e2)| (e1.clone() - e2.clone()).is_zero())
    }

    /// Retrieves the diagonal elements of the matrix and returns them as a column vector.
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
    ///     matrix::{matrix::Matrix, vector::VectorC},
    ///     number::instances::word8::Word8,
    /// };
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8, 2, 2>::of(&[1, 2, 4, 6].map(|e| Word8::of(e))).unwrap();
    ///     let d = m.get_diagonal();
    ///     let v1 = Word8::of(1);
    ///     let v2 = Word8::of(6);
    ///     assert_eq!(d, VectorC::<&Word8, 2>::of(&[&v1, &v2]).unwrap());
    /// }
    /// ```
    pub fn get_diagonal(&self) -> VectorC<&N, { Self::get_diagonal_length() }> {
        let length = Self::get_diagonal_length();
        let mut inner = Vec::with_capacity(length);
        for index in 1..=length {
            inner.push(&self[(index, index)]);
        }
        return VectorC { inner };
    }

    /// Retrieves the submatrix obtained by removing a specific row and column from the matrix.
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
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
    ///     let sub = m.submatrix(1, 2);
    ///     let n = Matrix::<Word8, 2, 2>::of(&[4, 6, 7, 9].map(|e| Word8::of(e))).unwrap();
    ///     assert_eq!(sub, n);
    /// }
    /// ```
    pub fn submatrix(&self, row: usize, column: usize) -> Matrix<N, { R - 1 }, { C - 1 }>
    where
        N: Clone,
    {
        let mut inner = Vec::with_capacity((R - 1) * (C - 1));
        for row_index in 1..=R {
            for column_index in 1..=C {
                if row_index == row || column_index == column {
                    continue;
                } else {
                    inner.push(self[(row_index, column_index)].clone());
                }
            }
        }
        Matrix { inner }
    }
}

/// Implements `PartialEq` for matrices with elements of types that implement `PartialEq`.
impl<N, const R1: usize, const C1: usize, const R2: usize, const C2: usize>
    std::cmp::PartialEq<Matrix<N, R2, C2>> for Matrix<N, R1, C1>
where
    N: std::cmp::PartialEq + std::marker::Sync,
{
    fn eq(&self, other: &Matrix<N, R2, C2>) -> bool {
        R1 == R2
            && C1 == C2
            && self
                .inner
                .par_iter()
                .zip(other.inner.par_iter())
                .all(|(e1, e2)| e1 == e2)
    }
}

/// Negates all elements of the matrix.
///
/// Note that for unsigned numbers, such as `Word`, negating will cause the value to wrap around,
/// i.e., -a = MAX - a.
impl<N, const R: usize, const C: usize> std::ops::Neg for Matrix<N, R, C>
where
    N: Number,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        let inner = self.inner.par_iter().map(|e1| -e1.clone()).collect();
        Self { inner }
    }
}

/// Calculates the matrix added to a scalar.
impl<N, const R: usize, const C: usize> std::ops::Add<N> for Matrix<N, R, C>
where
    N: Number,
{
    type Output = Self;

    fn add(self, rhs: N) -> Self::Output {
        let inner = self
            .inner
            .par_iter()
            .map(|e| e.clone() + rhs.clone())
            .collect();
        Self { inner }
    }
}

/// Calculates the matrix subtracted by a scalar.
impl<N, const R: usize, const C: usize> std::ops::Sub<N> for Matrix<N, R, C>
where
    N: Number,
{
    type Output = Self;

    fn sub(self, rhs: N) -> Self::Output {
        let inner = self
            .inner
            .par_iter()
            .map(|e| e.clone() - rhs.clone())
            .collect();
        Self { inner }
    }
}

/// Calculates the matrix multiplied by a scalar.
impl<N, const R: usize, const C: usize> std::ops::Mul<N> for Matrix<N, R, C>
where
    N: Number,
{
    type Output = Self;

    fn mul(self, rhs: N) -> Self::Output {
        let inner = self
            .inner
            .par_iter()
            .map(|e| e.clone() * rhs.clone())
            .collect();
        Self { inner }
    }
}

/// Calculates the matrix divided by a scalar.
impl<N, const R: usize, const C: usize> std::ops::Div<N> for Matrix<N, R, C>
where
    N: Fractional,
{
    type Output = Self;

    fn div(self, rhs: N) -> Self::Output {
        apply(&self, |e| e / rhs.clone())
    }
}

/// Calculates the matrix added to another matrix,
/// requiring that both matrices have the same shape.
impl<N, const R: usize, const C: usize> std::ops::Add for Matrix<N, R, C>
where
    N: Number,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let inner = self
            .inner
            .par_iter()
            .zip(rhs.inner.par_iter())
            .map(|(e1, e2)| e1.clone() + e2.clone())
            .collect();
        Self { inner }
    }
}

/// Calculates the matrix subtracted to another matrix,
/// requiring that both matrices have the same shape.
impl<N, const R: usize, const C: usize> std::ops::Sub for Matrix<N, R, C>
where
    N: Number,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let inner = self
            .inner
            .par_iter()
            .zip(rhs.inner.par_iter())
            .map(|(e1, e2)| e1.clone() - e2.clone())
            .collect();
        Self { inner }
    }
}

/// Calculates the matrix added to another matrix,
/// requiring that both matrices have complementary shapes.
impl<N, const R: usize, const K: usize, const C: usize> std::ops::Mul<Matrix<N, K, C>>
    for Matrix<N, R, K>
where
    N: Number,
{
    type Output = Matrix<N, R, C>;

    fn mul(self, rhs: Matrix<N, K, C>) -> Self::Output {
        (1..=K)
            .into_par_iter()
            .map(|k_index| {
                layer_product(
                    apply(
                        &self
                            .get_column(k_index)
                            .expect("Error[Matrix::mul]: k_index should be a valid index."),
                        |e| e.clone(),
                    ), // lhs[:, k]
                    apply(
                        &rhs.get_row(k_index)
                            .expect("Error[Matrix::mul]: k_index should be a valid index."),
                        |e| e.clone(),
                    ), // rhs[k, :]
                )
            })
            .reduce(|| Matrix::default(), |a, b| a + b)
    }
}

/// Indexes matrix elements using row and column coordinates.
impl<N, const R: usize, const C: usize> std::ops::Index<(usize, usize)> for Matrix<N, R, C> {
    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("Error[Matrix::index]: Index is out of bounds.")
    }
}

/// Obtains a mutable reference to matrix elements using row and column coordinates.
impl<N, const R: usize, const C: usize> std::ops::IndexMut<(usize, usize)> for Matrix<N, R, C> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.inner[Self::position_to_index(index.0, index.1)]
    }
}

/// Implements `Display` for matrices with elements of types that implement `Display`.
impl<N, const R: usize, const C: usize> std::fmt::Display for Matrix<N, R, C>
where
    N: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row_index in 1..=R {
            write!(f, "[")?;
            for column_index in 1..C {
                write!(f, "{}, ", self[(row_index, column_index)])?;
            }
            writeln!(f, "{}]", self[(row_index, C)])?;
        }
        Ok(())
    }
}

/// Implements `Debug` for matrices with elements of types that implement `Debug`.
impl<N, const R: usize, const C: usize> std::fmt::Debug for Matrix<N, R, C>
where
    N: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row_index in 1..=R {
            write!(f, "[")?;
            for column_index in 1..C {
                write!(f, "{:?}, ", self[(row_index, column_index)])?;
            }
            writeln!(f, "{:?}]", self[(row_index, C)])?;
        }
        Ok(())
    }
}

/// Implements `Default` for matrices with elements of types that implement `Clone` and `Default`.
impl<N, const R: usize, const C: usize> std::default::Default for Matrix<N, R, C>
where
    N: Clone + Default,
{
    fn default() -> Self {
        Self {
            inner: vec![N::default(); R * C],
        }
    }
}
