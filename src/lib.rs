pub trait MatrixLike {
    fn matrix_size(&self) -> (usize, usize);
    fn matrix_time(self, m: Box<dyn MatrixLike>) -> Option<Matrix>;
}

pub fn vec_pos(size: &(usize, usize), pos: &(usize, usize)) -> usize {
    (pos.0 - 1) * size.1 + pos.1
}

/// normal matrix with data and size
pub struct Matrix {
    /// data for matrix
    data: Vec<f64>,
    /// size for matrix
    size: (usize, usize),
}

impl MatrixLike for Matrix {
    fn matrix_size(&self) -> (usize, usize) {
        (self.size.0, self.size.1)
    }

    fn matrix_time<'a>(self, m: Box<dyn MatrixLike>) -> Option<Matrix> {
        let m_size = m.matrix_size();
        if m_size.0 != self.size.1 {
            None
        } else {
            None
        }
    }
}

impl Matrix {
    pub fn zeros(r: usize, c: usize) -> Self {
        //! return a zero matrix with specific size
        //! - size must greater than 0
        //!
        //! # Example
        //!
        //! ```
        //! # use rmatrix::Matrix;
        //! // get a 3x3 zero matrix
        //! let zero_matrix = Matrix::zeros(3, 3);
        //! ```
        Matrix {
            data: vec![0.0f64; r * c],
            size: (r, c),
        }
    }

    pub fn matrix_get_element(&self, pos: (usize, usize)) -> Option<f64> {
        if let Some(v) = self.data.get(vec_pos(&self.size, &pos)) {
            Some(*v)
        } else {
            None
        }
    }
}
