#[derive(Debug)]
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
            panic!("size must greater than 0");
        }
        MatrixSize { row, col }
    }

    pub fn into_vec_size(&self, position_row: usize, position_col: usize) -> usize {
        //! transform matrix position to vector postion
        //!
        //! # Examples
        //!
        //! ```rust
        //! # use rmatrix::matrix_size::MatrixSize;
        //! let v_pos = MatrixSize::new(2, 2).into_vec_size(1, 2);
        //! assert_eq!(1, v_pos);
        //! ```
        (position_row - 1) * self.col + position_col - 1
    }
}
