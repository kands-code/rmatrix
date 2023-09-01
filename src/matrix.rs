use crate::matrix_size::MatrixSize;

#[derive(Debug)]
/// normal matrix with data and size
pub struct Matrix {
    /// data for matrix
    data: Vec<f64>,
    /// size for matrix
    size: MatrixSize,
}

impl Matrix {
    pub fn from_vec(r: usize, c: usize, data: Vec<f64>) -> Self {
        //! init a matrix from a vec
        //!
        //! vec will fill the matrix by row
        //!
        //! # Examples
        //!
        //! ```rust
        //! # use rmatrix::matrix::Matrix;
        //! let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        //! println!("{}", m); // will print [1, 2]
        //!                    //            [3. 4]
        //! ```
        Matrix {
            data: data.split_at(r * c).0.to_owned(),
            size: MatrixSize::new(r, c),
        }
    }

    pub fn zeros(r: usize, c: usize) -> Self {
        //! return a zero matrix with specific size
        //!
        //! # Examples
        //!
        //! ```rust
        //! # use rmatrix::matrix::Matrix;
        //! // get a 3x3 zero matrix
        //! let zero_matrix = Matrix::zeros(3, 3);
        //! ```
        Matrix {
            data: vec![0.0f64; r * c],
            size: MatrixSize::new(r, c),
        }
    }

    pub fn matrix_get_element(&self, position_row: usize, position_col: usize) -> f64 {
        *self
            .data
            .get(self.size.into_vec_size(position_row, position_col))
            .unwrap()
    }

    pub fn matrix_time<'a>(self, m: Matrix) -> Option<Matrix> {
        if self.size.col != m.size.row {
            None
        } else {
            None
        }
    }
}

impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for r in 1..=self.size.row {
            write!(f, "[")?;
            for c in 1..=(self.size.col - 1) {
                write!(f, "{}, ", self.matrix_get_element(r, c))?;
            }
            write!(f, "{}]\n", self.matrix_get_element(r, self.size.col))?;
        }
        Ok(())
    }
}
