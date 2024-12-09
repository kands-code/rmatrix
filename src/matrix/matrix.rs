//! # Matrix
//!
//! Basic matrix type.

use super::vector::{VectorC, VectorR};

#[derive(Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde_mat", derive(serde::Deserialize, serde::Serialize))]
pub struct Matrix<N, const R: usize, const C: usize> {
    pub(crate) inner: Vec<N>,
}

impl<N, const R: usize, const C: usize> Matrix<N, R, C> {
    pub(crate) const fn index_to_position(row_index: usize, column_index: usize) -> usize {
        (row_index - 1) * C + column_index - 1
    }

    pub(crate) const fn get_diagonal_length() -> usize {
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
            Some(Matrix {
                inner: Vec::from(&data[..R * C]),
            })
        }
    }

    pub const fn dimension() -> (usize, usize) {
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
            self.inner.get(position)
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
                inner[column_index - 1] = &self[(row_index, column_index)];
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
                inner[row_index - 1] = &self[(row_index, column_index)];
            }
            Some(VectorC { inner })
        }
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
