//! # Error
//!
//! self define errors, for debug

/// errors what will happen with this library
#[derive(Debug, PartialEq)]
pub enum IError {
    /// likely to occur when constructing a matrix
    IncompatibleSizeError((usize, usize), usize),
    /// likely to occur when the function need a square matrix
    IncompatibleShape((usize, usize), (usize, usize)),
    /// likely to occur when getting or setting value of a matrix
    OutOfBoundary(usize, usize),
    /// likely to occur when dividing by ZERO
    DividedByZero,
    /// likely to occur when matrix is strange
    SingularMatrix,
    /// will occur when linear equations ROW > COL
    NoSolution((usize, usize), usize),
    /// likely occur when using serde_mat
    Message(String),
}

pub type IResult<T> = std::result::Result<T, IError>;

impl std::error::Error for IError {}

impl std::fmt::Display for IError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IError::IncompatibleSizeError((required_row, required_col), real) => write!(
                f,
                "incompatible size while require ({:?} x {:?}) but the real size is {:?}",
                required_row, required_col, real
            ),
            IError::IncompatibleShape(required, real) => {
                write!(
                    f,
                    "incompatible shape while require {:?} but the real shape is {:?}",
                    required, real
                )
            }
            IError::OutOfBoundary(row, col) => {
                write!(f, "out of boundary at ({:?}, {:?})", row, col)
            }
            IError::DividedByZero => write!(f, "can not divide by zero"),
            IError::SingularMatrix => write!(f, "matrix is strange which determinant is zero"),
            IError::NoSolution((row, col), rank) => {
                write!(
                    f,
                    "linear equation {:?}x{:?} with rank {:?} has no solution",
                    row, col, rank
                )
            }
            IError::Message(msg) => write!(f, "{:?}", msg),
        }
    }
}
