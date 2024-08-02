//! # Error
//!
//! self define errors, for debug

/// errors what will happen with this library
#[derive(Debug)]
pub enum MatrixError {
    /// likely to occur when constructing a matrix
    IncompatibleSizeError((usize, usize), usize),
    /// likely to occur when the function need a sqaure matrix
    IncompatibleShape((usize, usize), (usize, usize)),
    /// likely to occur when getting or setting value of a matrix
    OutOfBoundary(usize, usize),
    /// likely to occur when dividing by ZERO
    DividedByZero,
    /// likely to occur when read from string
    IncompatibleFormat(String, &'static str),
    /// will only occur when doing division with something like quaternion
    NoDivision(&'static str),
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
            MatrixError::DividedByZero => write!(f, "can not divide by zero"),
            MatrixError::IncompatibleFormat(from, type_name) => {
                write!(f, "can not read {:?} into {:?}", from, type_name)
            }
            MatrixError::NoDivision(type_name) => {
                write!(f, "no proper division can do with {:?}", type_name)
            }
        }
    }
}
