//! # Error
//!
//! self define errors, for debug

/// errors what will happen with this library
#[derive(Debug, PartialEq)]
pub enum MatrixError {
    /// likely to occur when constructing a matrix
    IncompatibleSizeError((usize, usize), usize),
    /// likely to occur when the function need a sqaure matrix
    IncompatibleShape((usize, usize), (usize, usize)),
    /// likely to occur when getting or setting value of a matrix
    OutOfBoundary(usize, usize),
    /// likely to occur when getting or setting value of a array/vector
    OutOfLength(usize, usize),
    /// likely to occur when dividing by ZERO
    DividedByZero,
    /// likely to occur when matrix is strange
    StrangeMatrix,
    /// will occur when linear equations ROW > COL
    NoSolution(usize, usize),
}

impl std::error::Error for MatrixError {}

impl std::fmt::Display for MatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixError::IncompatibleSizeError((required_row, required_col), real) => write!(
                f,
                "incompatible size while require ({:?} x {:?}) but the real size is {:?}",
                required_row, required_col, real
            ),
            MatrixError::IncompatibleShape(required, real) => {
                write!(
                    f,
                    "incompatible shape while require {:?} but the real shape is {:?}",
                    required, real
                )
            }
            MatrixError::OutOfBoundary(row, col) => {
                write!(f, "out of boundary at ({:?}, {:?})", row, col)
            }
            MatrixError::OutOfLength(length, index) => {
                write!(
                    f,
                    "length of array/vector is {:?} but index is {:?}",
                    length, index
                )
            }
            MatrixError::DividedByZero => write!(f, "can not divide by zero"),
            MatrixError::StrangeMatrix => write!(f, "matrix is strange which determinant is zero"),
            MatrixError::NoSolution(row, col) => {
                write!(f, "linear equation with {}x{} has no solution", row, col)
            }
        }
    }
}
