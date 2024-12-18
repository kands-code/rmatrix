//! # Matrix
//!
//! Basic matrix type.

use crate::{
    matrix::{
        utils::map,
        vector::{layer_product, VectorC, VectorR},
    },
    number::traits::{fractional::Fractional, number::Number},
};

#[cfg(feature = "rand_mat")]
use rand::distributions::{uniform::SampleUniform, Distribution, Uniform};

#[derive(Clone)]
pub struct Matrix<N, const R: usize, const C: usize> {
    pub(crate) inner: Vec<N>,
}

impl<N, const R: usize, const C: usize> Matrix<N, R, C> {
    pub(crate) fn index_to_position(row_index: usize, column_index: usize) -> usize {
        (row_index - 1) * C + column_index - 1
    }

    pub const fn get_diagonal_length() -> usize {
        if R > C {
            C
        } else {
            R
        }
    }

    pub fn of(data: &[N]) -> Option<Self>
    where
        N: Clone,
    {
        if data.len() < R * C {
            eprintln!(
                "Error[Matrix::of]: data length {} is less than matrix require {}",
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

    pub fn diagonal(data: &[N]) -> Option<Self>
    where
        N: Clone + Default,
    {
        let length = Self::get_diagonal_length();
        if data.len() < length {
            eprintln!(
                "Error[Matrix::diagonal]: data length {} is less than matrix require {}",
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

    pub fn p_muls(row: usize, scalar: N) -> Matrix<N, R, R>
    where
        N: Number,
    {
        let mut p_mat = Matrix::eyes();
        p_mat[(row, row)] = scalar;
        p_mat
    }

    pub fn p_add(from: usize, to: usize, scalar: N) -> Matrix<N, R, R>
    where
        N: Number,
    {
        let mut p_mat = Matrix::eyes();
        p_mat[(to, from)] = scalar;
        p_mat
    }

    #[cfg(feature = "rand_mat")]
    pub fn rand(lb: N, ub: N) -> Self
    where
        N: Number + SampleUniform,
    {
        let range = Uniform::new(lb, ub);
        let mut rng = rand::thread_rng();
        let inner = range.sample_iter(&mut rng).take(R * C).collect();
        Self { inner }
    }

    pub fn dimension(&self) -> (usize, usize) {
        (R, C)
    }

    pub fn linear_iter(&self) -> std::slice::Iter<'_, N> {
        self.inner.iter()
    }

    pub fn get(&self, row_index: usize, column_index: usize) -> Option<&N> {
        if row_index == 0 || column_index == 0 || row_index > R || column_index > C {
            eprintln!(
                "Error[Matrix::get]: index ({}, {}) is out of boundary",
                row_index, column_index
            );
            None
        } else {
            let position = Self::index_to_position(row_index, column_index);
            self.linear_iter().nth(position)
        }
    }

    pub fn set(&mut self, row_index: usize, column_index: usize, value: N) {
        if row_index == 0 || column_index == 0 || row_index > R || column_index > C {
            eprintln!(
                "Error[Matrix::set]: index ({}, {}) is out of boundary",
                row_index, column_index
            );
        } else {
            let position = Self::index_to_position(row_index, column_index);
            self.inner[position] = value;
        }
    }

    pub fn get_row(&self, row_index: usize) -> Option<VectorR<&N, C>> {
        if row_index == 0 || row_index > R {
            eprintln!(
                "Error[Matrix::get_row]: row index {} is out of boundary",
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

    pub fn get_column(&self, column_index: usize) -> Option<VectorC<&N, R>> {
        if column_index == 0 || column_index > C {
            eprintln!(
                "Error[Matrix::get_column]: column index {} is out of boundary",
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

    pub fn get_diagonal(&self) -> VectorC<&N, { Self::get_diagonal_length() }> {
        let length = Self::get_diagonal_length();
        let mut inner = Vec::with_capacity(length);
        for index in 1..=length {
            inner.push(&self[(index, index)]);
        }
        return VectorC { inner };
    }

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

impl<N, const R1: usize, const C1: usize, const R2: usize, const C2: usize>
    std::cmp::PartialEq<Matrix<N, R2, C2>> for Matrix<N, R1, C1>
where
    N: PartialEq,
{
    fn eq(&self, other: &Matrix<N, R2, C2>) -> bool {
        R1 == R2
            && C1 == C2
            && self
                .linear_iter()
                .take(R1 * C2)
                .zip(other.linear_iter().take(R2 * C1))
                .all(|(e1, e2)| e1 == e2)
    }
}

impl<N, const R: usize, const C: usize> std::ops::Neg for Matrix<N, R, C>
where
    N: Number,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        let inner = self.linear_iter().map(|e1| -e1.clone()).collect();
        Self { inner }
    }
}

impl<N, const R: usize, const C: usize> std::ops::Add<N> for Matrix<N, R, C>
where
    N: Number,
{
    type Output = Self;

    fn add(self, rhs: N) -> Self::Output {
        let inner = self
            .linear_iter()
            .map(|e| e.clone() + rhs.clone())
            .collect();
        Self { inner }
    }
}

impl<N, const R: usize, const C: usize> std::ops::Sub<N> for Matrix<N, R, C>
where
    N: Number,
{
    type Output = Self;

    fn sub(self, rhs: N) -> Self::Output {
        self + (-rhs)
    }
}

impl<N, const R: usize, const C: usize> std::ops::Add for Matrix<N, R, C>
where
    N: Number,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let inner = self
            .linear_iter()
            .zip(rhs.linear_iter())
            .map(|(e1, e2)| e1.clone() + e2.clone())
            .collect();
        Self { inner }
    }
}

impl<N, const R: usize, const C: usize> std::ops::Sub for Matrix<N, R, C>
where
    N: Number,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<N, const R: usize, const C: usize> std::ops::Mul<N> for Matrix<N, R, C>
where
    N: Number,
{
    type Output = Self;

    fn mul(self, rhs: N) -> Self::Output {
        let inner = self
            .linear_iter()
            .map(|e| e.clone() * rhs.clone())
            .collect();
        Self { inner }
    }
}

impl<N, const R: usize, const K: usize, const C: usize> std::ops::Mul<Matrix<N, K, C>>
    for Matrix<N, R, K>
where
    N: Number,
{
    type Output = Matrix<N, R, C>;

    fn mul(self, rhs: Matrix<N, K, C>) -> Self::Output {
        let mut product = Matrix::default();
        for k_index in 1..=K {
            let lhs_k_column = map(
                &self
                    .get_column(k_index)
                    .expect("Error[Matrix::mul]: k_index should be valid index"),
                |e| e.clone(),
            ); // lhs[:, k]
            let rhs_k_row = map(
                &rhs.get_row(k_index)
                    .expect("Error[Matrix::mul]: k_index should be valid index"),
                |e| e.clone(),
            ); // rhs[k, :]
            let layer = layer_product(lhs_k_column, rhs_k_row);
            product = product + layer;
        }
        product
    }
}

impl<N, const R: usize, const C: usize> std::ops::Div<N> for Matrix<N, R, C>
where
    N: Fractional,
{
    type Output = Self;

    fn div(self, rhs: N) -> Self::Output {
        map(&self, |e| e / rhs.clone())
    }
}

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

impl<N, const R: usize, const C: usize> std::ops::Index<(usize, usize)> for Matrix<N, R, C> {
    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        self.get(index.0, index.1)
            .expect("Error[Matrix::index]: index out of boundary")
    }
}

impl<N, const R: usize, const C: usize> std::ops::IndexMut<(usize, usize)> for Matrix<N, R, C> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.inner[Self::index_to_position(index.0, index.1)]
    }
}
