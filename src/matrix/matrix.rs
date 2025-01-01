//! # matrix::matrix
//!
//! Definition of the matrix along with related functions and implementations.

use rand::distributions::{Distribution, Uniform, uniform::SampleUniform};
use rayon::iter::{
    IndexedParallelIterator,
    IntoParallelIterator,
    IntoParallelRefIterator,
    ParallelIterator,
};

use crate::{
    matrix::{utils::apply, vector::layer_product},
    number::traits::{floating::Floating, fractional::Fractional, number::Number},
};

/// A matrix is a container of a single type that has two dimensions: rows and columns.
#[derive(Clone)]
pub struct Matrix<N> {
    /// Internal container.
    pub(crate) inner: Vec<N>,
    /// Number of rows in the matrix.
    pub row: usize,
    /// Number of columns in the matrix.
    pub column: usize,
}

impl<N> Matrix<N> {
    /// Construct the matrix by passing in data.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8>::of(2, 2, &[1, 2, 3, 4, 5, 6].map(|e| Word8::of(e))).unwrap();
    ///     // Data will be truncated when the length exceeds the number of matrix elements.
    ///     assert_eq!(
    ///         m.linear_iter().cloned().collect::<Vec<Word8>>(),
    ///         [1, 2, 3, 4].map(|e| Word8::of(e)).to_vec()
    ///     );
    ///     let n = Matrix::<Word8>::of(2, 2, &[1, 2, 3].map(|e| Word8::of(e)));
    ///     // Returns None when the data length is less than the number of matrix elements.
    ///     assert_eq!(n, None);
    /// }
    /// ```
    pub fn of(row: usize, column: usize, data: &[N]) -> Option<Self>
    where
        N: Clone,
    {
        if data.len() < row * column {
            eprintln!(
                concat!(
                    "Error[Matrix::of]: ",
                    "The data length ({}) does not meet the required number ",
                    "of elements ({}) for the matrix."
                ),
                data.len(),
                row * column
            );
            None
        } else {
            Some(Self {
                inner: Vec::from(&data[..(row * column)]),
                row,
                column,
            })
        }
    }

    /// Construct a matrix of the specified shape filled with default values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let z = Matrix::<Word8>::defaults(2, 2);
    ///     let z_expect = Matrix::<Word8>::of(2, 2, &[0, 0, 0, 0].map(Word8::of)).unwrap();
    ///     assert_eq!(z, z_expect);
    /// }
    /// ```
    pub fn defaults(row: usize, column: usize) -> Self
    where
        N: Clone + Default,
    {
        Self {
            inner: vec![N::default(); row * column],
            row,
            column,
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
    ///     let i = Matrix::<Word8>::eyes(2, 2);
    ///     let m = Matrix::<Word8>::of(2, 2, &[1, 0, 0, 1].map(|e| Word8::of(e))).unwrap();
    ///     assert_eq!(i, m);
    /// }
    /// ```
    pub fn eyes(row: usize, column: usize) -> Self
    where
        N: Number,
    {
        let mut id_mat = Self::defaults(row, column);
        for index in 1..=id_mat.edge() {
            id_mat[(index, index)] = N::one();
        }
        id_mat
    }

    /// Constructing a matrix where all elements are filled with the given value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::fills(2, 2, Word8::of(2));
    ///     let m_expect = Matrix::of(2, 2, &[2, 2, 2, 2].map(Word8::of)).unwrap();
    ///     assert_eq!(m, m_expect);
    /// }
    /// ```
    pub fn fills(row: usize, column: usize, data: N) -> Self
    where
        N: Clone,
    {
        Self {
            inner: vec![data; row * column],
            row,
            column,
        }
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
    ///     let m = Matrix::<Word8>::diagonal(2, 3, &[1, 2].map(|e| Word8::of(e))).unwrap();
    ///     let d = Matrix::<Word8>::of(2, 3, &[1, 0, 0, 0, 2, 0].map(|e| Word8::of(e))).unwrap();
    ///     assert_eq!(m, d);
    ///     // Returns None when the data length is less than the length of the matrix diagonal.
    ///     let n = Matrix::<Word8>::of(2, 2, &[1].map(|e| Word8::of(e)));
    ///     assert_eq!(n, None);
    /// }
    /// ```
    pub fn diagonal(row: usize, column: usize, data: &[N]) -> Option<Self>
    where
        N: Clone + Default,
    {
        let length = row.min(column);
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
            let mut diag = Self::defaults(row, column);
            for index in 1..=length {
                diag[(index, index)] = data[index - 1].clone();
            }
            Some(diag)
        }
    }

    /// Construct the Vandermonde matrix.
    ///
    /// Where `row` is the length of the data,
    /// and assuming the order of the data is `n`, then `column = n + 1`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::double::Double};
    ///
    /// fn main() {
    ///     let m = Matrix::<Double>::vandermonde(5, 2, &[1.0, 2.0, 4.0, 6.0, 8.0].map(Double::of))
    ///         .unwrap();
    ///     let m_expect = Matrix::<Double>::of(
    ///         5,
    ///         2,
    ///         &[1.0, 1.0, 1.0, 2.0, 1.0, 4.0, 1.0, 6.0, 1.0, 8.0].map(Double::of),
    ///     )
    ///     .unwrap();
    ///     assert_eq!(m, m_expect);
    /// }
    /// ```
    pub fn vandermonde(row: usize, column: usize, data: &[N]) -> Option<Self>
    where
        N: Floating,
    {
        if data.len() < row {
            eprintln!(
                concat!(
                    "Error[Matrix::vandermonde]: ",
                    "Row ({}) should not exceed the length of the data ({})."
                ),
                row,
                data.len()
            );
            None
        } else {
            Some(Self {
                inner: data
                    .iter()
                    .map(|e| {
                        (0..column)
                            .map(|p| {
                                e.clone().power(N::from_str(&format!("{:?}", p)).expect(
                                    "Error[Matrix::vandermonde]: Failed to convert from usize.",
                                ))
                            })
                            .collect::<Vec<N>>()
                    })
                    .flatten()
                    .collect::<Vec<N>>(),
                row,
                column,
            })
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
    ///     let m = Matrix::<Word8>::of(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e)))
    ///         .unwrap();
    ///     let p = Matrix::<Word8>::p_change(3, 3, 1, 3);
    ///     let n = Matrix::<Word8>::of(3, 3, &[7, 8, 9, 4, 5, 6, 1, 2, 3].map(|e| Word8::of(e)))
    ///         .unwrap();
    ///     assert_eq!(p * m, n);
    /// }
    /// ```
    pub fn p_change(row: usize, column: usize, from: usize, to: usize) -> Self
    where
        N: Number,
    {
        let mut p_mat = Self::eyes(row, column);
        p_mat[(from, from)] = N::zero();
        p_mat[(to, to)] = N::zero();
        p_mat[(from, to)] = N::one();
        p_mat[(to, from)] = N::one();
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
    ///     let m = Matrix::<Word8>::of(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e)))
    ///         .unwrap();
    ///     let p = Matrix::<Word8>::p_muls(3, 3, 2, Word8::of(2));
    ///     let n = Matrix::<Word8>::of(3, 3, &[1, 2, 3, 8, 10, 12, 7, 8, 9].map(|e| Word8::of(e)))
    ///         .unwrap();
    ///     assert_eq!(p * m, n);
    /// }
    /// ```
    pub fn p_muls(row: usize, column: usize, with: usize, scalar: N) -> Self
    where
        N: Number,
    {
        let mut p_mat = Self::eyes(row, column);
        p_mat[(with, with)] = scalar;
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
    ///     let m = Matrix::<Int8>::of(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Int8::of(e)))
    ///         .unwrap();
    ///     let p = Matrix::<Int8>::p_add(3, 3, 1, 2, Int8::of(-4));
    ///     let n = Matrix::<Int8>::of(3, 3, &[1, 2, 3, 0, -3, -6, 7, 8, 9].map(|e| Int8::of(e)))
    ///         .unwrap();
    ///     assert_eq!(p * m, n);
    /// }
    /// ```
    pub fn p_add(row: usize, column: usize, from: usize, to: usize, scalar: N) -> Self
    where
        N: Number,
    {
        let mut p_mat = Matrix::eyes(row, column);
        p_mat[(to, from)] = scalar;
        p_mat
    }

    /// Constructs a matrix of a specific shape, where each element is randomly selected from a given range.
    ///
    /// The `rand` library is used, and the matrix element type must implement the `SampleUniform` trait.
    /// In particular, the `Ratio` type cannot be used.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::float::Float};
    ///
    /// fn main() {
    ///     let m = Matrix::<Float>::rand(3, 3, Float::of(-1.0), Float::of(3.0));
    ///     assert!(
    ///         m.linear_iter()
    ///             .all(|e| Float::of(-1.0) < e.clone() && e.clone() < Float::of(3.0))
    ///     );
    /// }
    /// ```
    ///
    /// ## Warnings
    ///
    /// <div class="warning">
    ///
    /// For complex numbers, the ranges for the elements are defined by the real and imaginary parts.
    /// For example, if `lb` is passed as `Complex(-1.0, -3.0)` and `ub` as `Complex(3.0, 1.0)`,
    /// the real part of the matrix elements will be randomly selected from the range `[-1.0, 3.0)`,
    /// while the imaginary part will be randomly selected from the range `[-3.0, 1.0)`.
    ///
    /// </div>
    pub fn rand(row: usize, column: usize, lb: N, ub: N) -> Self
    where
        N: Number + SampleUniform,
    {
        let range = Uniform::new(lb, ub);
        let mut rng = rand::thread_rng();
        let inner = range.sample_iter(&mut rng).take(row * column).collect();
        Self { inner, row, column }
    }

    /// Convert the matrix element positions to internal container indices.
    pub(crate) fn position_to_index(
        column: usize,
        row_index: usize,
        column_index: usize,
    ) -> usize {
        (row_index - 1) * column + column_index - 1
    }

    /// Returns the shape of the matrix, specifically the number of rows and columns.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8>::of(2, 3, &[1, 2, 3, 4, 5, 6].map(|e| Word8::of(e))).unwrap();
    ///     assert_eq!(m.shape(), (2, 3));
    /// }
    /// ```
    pub fn shape(&self) -> (usize, usize) { (self.row, self.column) }

    /// Get the length of the matrix diagonal.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8>::eyes(3, 2);
    ///     assert_eq!(m.edge(), 2);
    /// }
    /// ```
    pub const fn edge(&self) -> usize {
        if self.row > self.column {
            self.column
        } else {
            self.row
        }
    }

    /// Returns the internal data of the matrix as an iterator.
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let data = [1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e));
    ///     let m = Matrix::<Word8>::of(3, 3, &data).unwrap();
    ///     assert!(m.linear_iter().zip(data.iter()).all(|(e1, e2)| e1 == e2));
    /// }
    /// ```
    pub fn linear_iter(&self) -> std::slice::Iter<'_, N> { self.inner.iter() }

    /// Retrieves the element at a specific position in the matrix.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8>::of(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e)))
    ///         .unwrap();
    ///     assert_eq!(m.get(2, 1), Some(&Word8::of(4)));
    ///     assert_eq!(m.get(3, 2), Some(&Word8::of(8)));
    ///     // Returns `None` when the position exceeds the boundaries.
    ///     assert_eq!(m.get(4, 2), None);
    /// }
    /// ```
    pub fn get(&self, row_index: usize, column_index: usize) -> Option<&N> {
        if row_index == 0
            || column_index == 0
            || row_index > self.row
            || column_index > self.column
        {
            eprintln!(
                "Error[Matrix::get]: Index ({}, {}) is out of bounds.",
                row_index, column_index
            );
            None
        } else {
            let position = Self::position_to_index(self.column, row_index, column_index);
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
    ///         Matrix::<Word8>::of(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e)))
    ///             .unwrap();
    ///     m.set(2, 1, Word8::of(12));
    ///     m.set(3, 2, Word8::of(16));
    ///     let n = Matrix::<Word8>::of(3, 3, &[1, 2, 3, 12, 5, 6, 7, 16, 9].map(|e| Word8::of(e)))
    ///         .unwrap();
    ///     assert_eq!(m, n);
    /// }
    /// ```
    pub fn set(&mut self, row_index: usize, column_index: usize, value: N) {
        if row_index == 0
            || column_index == 0
            || row_index > self.row
            || column_index > self.column
        {
            eprintln!(
                "Error[Matrix::set]: Index ({}, {}) is out of bounds.",
                row_index, column_index
            );
        } else {
            let position = Self::position_to_index(self.column, row_index, column_index);
            self.inner[position] = value;
        }
    }

    /// Retrieves a specific row of the matrix and returns it as a row vector.
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
    ///     let m = Matrix::<Word8>::of(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e)))
    ///         .unwrap();
    ///     // What is retrieved is a reference to the element, not the value of the element.
    ///     let r2 = m.get_row(2).unwrap();
    ///     let v1 = Word8::of(4);
    ///     let v2 = Word8::of(5);
    ///     let v3 = Word8::of(6);
    ///     let r2_expect = row_vector::<&Word8>(3, &[&v1, &v2, &v3]).unwrap();
    ///     assert_eq!(r2, r2_expect);
    ///     // Returns `None` when the position exceeds the boundaries.
    ///     let n = m.get_row(4);
    ///     assert_eq!(n, None);
    /// }
    /// ```
    pub fn get_row(&self, row_index: usize) -> Option<Matrix<&N>> {
        if row_index == 0 || row_index > self.row {
            eprintln!(
                "Error[Matrix::get_row]: Index ({}) is out of bounds.",
                row_index
            );
            None
        } else {
            let mut inner = Vec::with_capacity(self.column);
            for column_index in 1..=self.column {
                inner.push(&self[(row_index, column_index)]);
            }
            Some(Matrix {
                inner,
                row: 1,
                column: self.column,
            })
        }
    }

    /// Retrieves a specific column of the matrix and returns it as a column vector.
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
    ///     let m = Matrix::<Word8>::of(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e)))
    ///         .unwrap();
    ///     // What is retrieved is a reference to the element, not the value of the element.
    ///     let c2 = m.get_column(2).unwrap();
    ///     let v1 = Word8::of(2);
    ///     let v2 = Word8::of(5);
    ///     let v3 = Word8::of(8);
    ///     let c2_expect = column_vector::<&Word8>(3, &[&v1, &v2, &v3]).unwrap();
    ///     assert_eq!(c2, c2_expect);
    ///     // Returns `None` when the position exceeds the boundaries.
    ///     let n = m.get_column(4);
    ///     assert_eq!(n, None);
    /// }
    /// ```
    pub fn get_column(&self, column_index: usize) -> Option<Matrix<&N>> {
        if column_index == 0 || column_index > self.column {
            eprintln!(
                "Error[Matrix::get_column]: Index ({}) is out of bounds.",
                column_index
            );
            None
        } else {
            let mut inner = Vec::with_capacity(self.row);
            for row_index in 1..=self.row {
                inner.push(&self[(row_index, column_index)]);
            }
            Some(Matrix {
                inner,
                row: self.row,
                column: 1,
            })
        }
    }

    /// Retrieves the diagonal elements of the matrix and returns them as a column vector.
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
    ///     let m = Matrix::<Word8>::of(2, 2, &[1, 2, 4, 6].map(|e| Word8::of(e))).unwrap();
    ///     let d = m.get_diagonal();
    ///     let v1 = Word8::of(1);
    ///     let v2 = Word8::of(6);
    ///     assert_eq!(d, column_vector::<&Word8>(2, &[&v1, &v2]).unwrap());
    /// }
    /// ```
    pub fn get_diagonal(&self) -> Matrix<&N> {
        let length = self.edge();
        let mut inner = Vec::with_capacity(length);
        for index in 1..=length {
            inner.push(&self[(index, index)]);
        }
        return Matrix {
            inner,
            row: self.row,
            column: 1,
        };
    }

    /// Retrieves the submatrix obtained by removing a specific row and column from the matrix.
    ///
    /// # Panics
    ///
    /// Submatrices cannot be obtained for matrices with fewer than 2 rows or columns.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rmatrix_ks::{matrix::matrix::Matrix, number::instances::word8::Word8};
    ///
    /// fn main() {
    ///     let m = Matrix::<Word8>::of(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e)))
    ///         .unwrap();
    ///     let sub = m.submatrix(1, 2);
    ///     let n = Matrix::<Word8>::of(2, 2, &[4, 6, 7, 9].map(|e| Word8::of(e))).unwrap();
    ///     assert_eq!(sub, n);
    /// }
    /// ```
    pub fn submatrix(&self, row: usize, column: usize) -> Self
    where
        N: Clone,
    {
        assert!(
            self.row >= 2 && self.column >= 2,
            concat!(
                "Error[Matrix::submatrix]: ",
                "The matrix is too small to have any submatrices."
            )
        );

        let mut inner = Vec::with_capacity((self.row) * (self.column));
        for row_index in 1..=self.row {
            for column_index in 1..=self.column {
                if row_index == row || column_index == column {
                    continue;
                } else {
                    inner.push(self[(row_index, column_index)].clone());
                }
            }
        }
        Matrix {
            inner,
            row: self.row - 1,
            column: self.column - 1,
        }
    }
}

/// Implements `PartialEq` for matrices with elements of types that implement `PartialEq`.
impl<N> std::cmp::PartialEq<Matrix<N>> for Matrix<N>
where
    N: std::cmp::PartialEq + std::marker::Sync,
{
    fn eq(&self, other: &Self) -> bool {
        self.shape() == other.shape()
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
impl<N> std::ops::Neg for Matrix<N>
where
    N: Number,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        let inner = self.inner.par_iter().map(|e1| -e1.clone()).collect();
        Self {
            inner,
            row: self.row,
            column: self.column,
        }
    }
}

/// Calculates the matrix added to a scalar.
impl<N> std::ops::Add<N> for Matrix<N>
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
        Self {
            inner,
            row: self.row,
            column: self.column,
        }
    }
}

/// Calculates the matrix subtracted by a scalar.
impl<N> std::ops::Sub<N> for Matrix<N>
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
        Self {
            inner,
            row: self.row,
            column: self.column,
        }
    }
}

/// Calculates the matrix multiplied by a scalar.
impl<N> std::ops::Mul<N> for Matrix<N>
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
        Self {
            inner,
            row: self.row,
            column: self.column,
        }
    }
}

/// Calculates the matrix divided by a scalar.
impl<N> std::ops::Div<N> for Matrix<N>
where
    N: Fractional,
{
    type Output = Self;

    fn div(self, rhs: N) -> Self::Output {
        let inner = self
            .inner
            .par_iter()
            .map(|e| e.clone() / rhs.clone())
            .collect();
        Self {
            inner,
            row: self.row,
            column: self.column,
        }
    }
}

/// Calculates the matrix added to another matrix,
/// requiring that both matrices have the same shape.
impl<N> std::ops::Add for Matrix<N>
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
        Self {
            inner,
            row: self.row,
            column: rhs.column,
        }
    }
}

/// Calculates the matrix subtracted to another matrix,
/// requiring that both matrices have the same shape.
impl<N> std::ops::Sub for Matrix<N>
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
        Self {
            inner,
            row: self.row,
            column: rhs.column,
        }
    }
}

/// Calculates the matrix added to another matrix,
/// requiring that both matrices have complementary shapes.
impl<N> std::ops::Mul<Matrix<N>> for Matrix<N>
where
    N: Number,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.column, rhs.row,
            concat!(
                "Error[Matrix::mul]: ",
                "In matrix multiplication, the number of columns ({}) in the first matrix ",
                "must be equal to the number of rows ({}) in the second matrix."
            ),
            self.column, rhs.row
        );

        (1..=self.column)
            .into_par_iter()
            .map(|k_index| {
                layer_product(
                    &apply(
                        &self.get_column(k_index).expect(&format!(
                            concat!(
                                "Error[Matrix::mul]: ",
                                "k_index ({}) should be a valid index for self."
                            ),
                            k_index
                        )),
                        |e| e.clone(),
                    ), // lhs[:, k]
                    &apply(
                        &rhs.get_row(k_index).expect(&format!(
                            concat!(
                                "Error[Matrix::mul]: ",
                                "k_index ({}) should be a valid index for rhs."
                            ),
                            k_index
                        )),
                        |e| e.clone(),
                    ), // rhs[k, :]
                )
            })
            .reduce(|| Matrix::defaults(self.row, rhs.column), |a, b| a + b)
    }
}

/// Indexes matrix elements using row and column coordinates.
impl<N> std::ops::Index<(usize, usize)> for Matrix<N> {
    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("Error[Matrix::index]: Index is out of bounds.")
    }
}

/// Obtains a mutable reference to matrix elements using row and column coordinates.
impl<N> std::ops::IndexMut<(usize, usize)> for Matrix<N> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.inner[Self::position_to_index(self.column, index.0, index.1)]
    }
}

/// Implements `Display` for matrices with elements of types that implement `Display`.
impl<N> std::fmt::Display for Matrix<N>
where
    N: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row_index in 1..=self.row {
            write!(f, "[")?;
            for column_index in 1..self.column {
                write!(f, "{}, ", self[(row_index, column_index)])?;
            }
            writeln!(f, "{}]", self[(row_index, self.column)])?;
        }
        Ok(())
    }
}

/// Implements `Debug` for matrices with elements of types that implement `Debug`.
impl<N> std::fmt::Debug for Matrix<N>
where
    N: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row_index in 1..=self.row {
            write!(f, "[")?;
            for column_index in 1..self.column {
                write!(f, "{:?}, ", self[(row_index, column_index)])?;
            }
            writeln!(f, "{:?}]", self[(row_index, self.column)])?;
        }
        Ok(())
    }
}
