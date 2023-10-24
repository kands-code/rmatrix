#[derive(Debug, PartialEq)]
/// size of a matrix
pub struct MatrixSize {
    /// row size
    pub row: usize,
    /// column size
    pub col: usize,
}

impl MatrixSize {
    pub fn new(row: usize, col: usize) -> Self {
        //! init size for a matrix
        //!
        //! # Examples
        //!
        //! ```rust
        //! # use rmatrix::matrix_size::MatrixSize;
        //! // get a matrix size of 3 by 3
        //! let m_size = MatrixSize::new(3, 3);
        //! ```
        //!
        //! # Panics
        //!
        //! panic if r == 0 or c == 0
        //!
        //! ```rust,should_panic
        //! # use rmatrix::matrix_size::MatrixSize;
        //! // will panic
        //! MatrixSize::new(0, 2);
        //! ```
        if row == 0 || col == 0 {
            panic!("error: size must greater than 0");
        }
        MatrixSize { row, col }
    }

    fn into_vec_size(&self, position_row: usize, position_col: usize) -> usize {
        //! transform matrix position to vector postion
        //!
        //! # Examples
        //!
        //! ```rust
        //! # use rmatrix::matrix_size::MatrixSize;
        //! let v_pos = MatrixSize::new(2, 2).into_vec_size(1, 2);
        //! assert_eq!(1, v_pos);
        //! ```
        //!
        //! # Panics
        //!
        //! panic if position is out of boundary
        //!
        //! ```rust,should_panic
        //! # use rmatrix::matrix_size::MatrixSize;
        //! let _ = MatrixSize::new(2, 2).into_vec_size(3, 2);
        //! ```
        let v_size = (position_row - 1) * self.col + position_col - 1;
        if v_size < self.row * self.col {
            v_size
        } else {
            panic!(
                "error: ({}, {}) out of boundary!",
                position_row, position_col
            );
        }
    }
}

impl Default for MatrixSize {
    fn default() -> Self {
        Self { row: 1, col: 1 }
    }
}
