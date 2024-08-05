//! # Error
//!
//! self define errors, for debug

/// errors what will happen with this library
#[derive(Debug, PartialEq)]
pub enum Error {
    /// likely to occur when constructing a matrix
    IncompatibleSizeError((usize, usize), usize),
    /// likely to occur when the function need a sqaure matrix
    IncompatibleShape((usize, usize), (usize, usize)),
    /// likely to occur when getting or setting value of a matrix
    OutOfBoundary(usize, usize),
    /// likely to occur when dividing by ZERO
    DividedByZero,
    /// likely to occur when matrix is strange
    SingularMatrix,
    /// will occur when linear equations ROW > COL
    NoSolution(usize, usize),
    /// likely occur when using serde_mat
    Message(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::IncompatibleSizeError((required_row, required_col), real) => write!(
                f,
                "incompatible size while require ({:?} x {:?}) but the real size is {:?}",
                required_row, required_col, real
            ),
            Error::IncompatibleShape(required, real) => {
                write!(
                    f,
                    "incompatible shape while require {:?} but the real shape is {:?}",
                    required, real
                )
            }
            Error::OutOfBoundary(row, col) => {
                write!(f, "out of boundary at ({:?}, {:?})", row, col)
            }
            Error::DividedByZero => write!(f, "can not divide by zero"),
            Error::SingularMatrix => write!(f, "matrix is strange which determinant is zero"),
            Error::NoSolution(row, col) => {
                write!(f, "linear equation {:?}x{:?} has no solution", row, col)
            }
            Error::Message(msg) => write!(f, "{:?}", msg),
        }
    }
}
