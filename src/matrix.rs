use crate::matrix_size::MatrixSize;
use base64::{engine::general_purpose, Engine as _};

#[derive(Debug)]
/// normal matrix with data and size
pub struct Matrix {
    /// data for matrix
    data: Vec<f64>,
    /// size for matrix
    size: MatrixSize,
    /// name for matrix
    name: String,
}

impl Matrix {
    pub fn from_vec(row: usize, col: usize, data: Vec<f64>, name: Option<String>) -> Self {
        //! init a matrix from a vec
        //!
        //! vec will fill the matrix by row
        //!
        //! # Examples
        //!
        //! ```rust
        //! # use rmatrix::matrix::Matrix;
        //! let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0], None);
        //! println!("{}", m);
        //! ```
        let mut new_matrix = Matrix::default();
        new_matrix.data = data;
        new_matrix.size = MatrixSize::new(row, col);
        if let Some(n) = name {
            new_matrix.name = n;
        }
        new_matrix
    }

    pub fn zeros(row: usize, col: usize) -> Self {
        //! return a zero matrix with specific size
        //!
        //! # Examples
        //!
        //! ```rust
        //! # use rmatrix::matrix::Matrix;
        //! // get a 3x3 zero matrix
        //! let zero_matrix = Matrix::zeros(3, 3);
        //! ```
        Matrix::from_vec(row, col, vec![0.0f64; row * col], None)
    }

    pub fn matrix_get_element(&self, position_row: usize, position_col: usize) -> Option<&f64> {
        self.data
            .get(self.size.into_vec_size(position_row, position_col))
    }

    pub fn matrix_set_element(&mut self, elem: f64, position_row: usize, position_col: usize) {
        self.data[self.size.into_vec_size(position_row, position_col)] = elem;
    }

    pub fn matrix_add(self, m: Matrix) -> Option<Matrix> {
        if self.size != m.size {
            None
        } else {
            let mut new_matrix_data = Vec::with_capacity(self.size.row * m.size.col);
            for i in 0..self.size.row * m.size.col {
                new_matrix_data[i] = self.data[i] + m.data[i];
            }
            Some(Matrix::from_vec(
                m.size.row,
                self.size.col,
                new_matrix_data,
                None,
            ))
        }
    }

    pub fn matrix_mul(self, m: Matrix) -> Option<Matrix> {
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
                write!(f, "{:>6.3}, ", self.matrix_get_element(r, c).unwrap())?;
            }
            writeln!(
                f,
                "{:>6.3}]",
                self.matrix_get_element(r, self.size.col).unwrap()
            )?;
        }
        writeln!(
            f,
            "<<mat: {} {}x{}>>",
            self.name, self.size.row, self.size.col
        )
    }
}

impl Default for Matrix {
    fn default() -> Self {
        //! get default matrix
        //!
        //! by default:
        //! - the matrix data is empty
        //! - the matrix size is `MatrixSize::default()`
        //! - the matrix name is get from system time
        //!
        //! the base64 encode of time is normally
        //!
        //! `AAAAAAAAAAAAAAGKUdRxFA==`
        //!
        //! so the name is the range `14..21`, means`"GKUdRxF"`
        //!
        //! # Examples
        //!
        //! ```rust
        //! # use rmatrix::matrix::Matrix;
        //! let _ = Matrix::default();
        //! ```
        let mut rand_id = general_purpose::STANDARD.encode(
            (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap())
            .as_millis()
            .to_be_bytes(),
        );
        if rand_id.len() > 7 {
            rand_id = rand_id[14..21].to_owned();
        }
        Matrix {
            data: Vec::new(),
            size: MatrixSize::default(),
            name: format!("{}", rand_id),
        }
    }
}
