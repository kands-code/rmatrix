#[derive(Debug, Clone)]
pub enum RMatrixError {
    ShapeUnreasonable,
    ParseFailed(String),
    DivideByZero,
    OutOfBoundary(usize, usize),
    OutOfRowBoundary(usize),
    OutOfColumnBoundary(usize),
    LengthInconsistent(usize, usize),
    ShapeInconsistent((usize, usize), (usize, usize)),
}

impl std::fmt::Display for RMatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RMatrixError::ShapeUnreasonable => write!(f, "shape should be at least (1, 1)"),
            RMatrixError::ParseFailed(s) => write!(f, "parse failed with {}", s),
            RMatrixError::DivideByZero => write!(f, "divide by zero"),
            RMatrixError::OutOfBoundary(r, c) => {
                write!(f, "position ({}, {}) out of boundary!", r, c)
            }
            RMatrixError::OutOfRowBoundary(r) => write!(f, "row {} is out of boundary", r),
            RMatrixError::OutOfColumnBoundary(c) => {
                write!(f, "column {} is out of boundary", c)
            }
            RMatrixError::LengthInconsistent(l1, l2) => {
                write!(f, "vector length {} is inconsistent with from {}", l1, l2)
            }
            RMatrixError::ShapeInconsistent(s1, s2) => {
                write!(
                    f,
                    "shape {}x{} is inconsistent with shape {}x{}",
                    s1.0, s1.1, s2.0, s2.1
                )
            }
        }
    }
}

impl std::error::Error for RMatrixError {}
