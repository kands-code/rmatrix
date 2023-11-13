use crate::{
    error::RMatrixError,
    matrix::{shape::MatrixShape, Matrix, TAG_LEANGTH},
    number::Number,
};
use rand::{distributions::Alphanumeric, thread_rng, Rng};

impl<N: Number> Matrix<N> {
    /// return a zero matrix with specific size
    ///
    /// # Examples
    ///
    /// ```
    /// # use rmatrix::matrix::Matrix;
    /// # use rmatrix::error::RMatrixError;
    /// // get a 3x3 zero matrix
    /// # fn main() -> Result<(), RMatrixError> {
    /// let zero_matrix = Matrix::<f64>::zeros(3, 3)?;
    /// #     Ok(())  
    /// # }
    /// ```
    pub fn zeros(r: usize, c: usize) -> Result<Self, RMatrixError> {
        fn random_tag(n: usize) -> String {
            thread_rng()
                .sample_iter(&Alphanumeric)
                .take(n)
                .map(char::from)
                .collect()
        }

        Ok(Matrix {
            shape: MatrixShape::new(r, c)?,
            data: vec![Default::default(); r * c],
            tag: random_tag(TAG_LEANGTH),
        })
    }

    pub fn eyes(r: usize, c: usize) -> Result<Matrix<N>, RMatrixError> {
        let mut m = Self::zeros(r, c)?;
        for i in 1..=r.min(c) {
            m.set(N::one(), i, i)?;
        }
        Ok(m)
    }

    pub fn from_vec(row: usize, col: usize, data: Vec<N>) -> Result<Self, RMatrixError> {
        //! init a matrix from a vector
        //!
        //! vector will fill the matrix by row
        //!
        //! # Examples
        //!
        //! ```rust
        //! # use rmatrix::matrix::Matrix;
        //! let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        //! println!("{}", m);
        //! // will show like this:
        //! // [ 1.000,  2.000]
        //! // [ 3.000,  4.000]
        //! // <mat[f8YWn3K7] 2x2>
        //! ```

        let mut m = Self::zeros(row, col)?;
        m.data = data;
        Ok(m)
    }

    pub fn from_stdin() -> Result<Self, RMatrixError> {
        //! read a matrix from stdin
        //!
        //! # Examples
        //!
        //! the input must follow some rules
        //!
        //! - the shape infos are separated by spaces or commas
        //! - the shape info must in one line
        //! - the data are separated by spaces or newlines
        //! - can not use comma or other separator with data
        //!
        //! ```no_run
        //! # use rmatrix::matrix::Matrix;
        //! let m = Matrix::<f64>::from_stdin().unwrap();
        //! println!("{}", m);
        //!
        //! // matrix shape (r, c): 2, 2    
        //! // matrix data:
        //! // 1 2
        //! // 3 4.15
        //! //
        //! // [ 1.000,  2.000]
        //! // [ 3.000,  4.150]
        //! // <mat[c2i8szSj] 2x2>
        //! ```
        //!
        //! # Warning
        //!
        //! for complex number of other type
        //!
        //! `from_stdin()` method is using whitespace to split data,
        //! be sure to avoid whitespace when entering
        //!
        //! if the complex number has only an `Im`, the `Re` should be replaced by 0,
        //! but there must be a `Re`.
        //!
        //! ## Examples
        //!
        //! ```no_run
        //! # use rmatrix::matrix::Matrix;
        //! # use rmatrix::complex::Complex;
        //! let m = Matrix::<Complex>::from_stdin().unwrap();
        //! println!("{}", m);
        //!
        //! // matrix shape (r, c): 2 2
        //! // matrix data:
        //! // 1+2I -3-3.14I
        //! // 0-4I -3.14+2I
        //! //
        //! // [1.000+2.000I, -3.000-3.140I]
        //! // [0.000-4.000I, -3.140+2.000I]
        //! // <mat[nhQSomO8] 2x2>
        //! ```

        print!("matrix shape (r, c): ");
        std::io::Write::flush(&mut std::io::stdout()).expect("failed to flush stdout");
        let mut rb: String = String::new();
        let mut shape_info: Vec<usize>;
        loop {
            rb.clear();
            if std::io::stdin().read_line(&mut rb).is_ok() {
                shape_info = rb
                    .split(|c: char| c.is_whitespace() || c == ',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<usize>().unwrap_or_default())
                    .collect::<Vec<_>>();
            } else {
                loop {
                    eprintln!("failed to get matrix shape info, please re-input!");
                    if let Ok(_) = std::io::stdin().read_line(&mut rb) {
                        shape_info = rb
                            .split(|c: char| c.is_whitespace() || c == ',')
                            .filter(|s| !s.is_empty())
                            .map(|s| s.parse::<usize>().unwrap_or_default())
                            .collect::<Vec<_>>();
                        break;
                    }
                }
            }

            if shape_info.len() == 2 {
                break;
            } else {
                eprintln!("failed to get matrix shape info, please re-input!");
                print!("matrix shape (r, c): ");
                std::io::Write::flush(&mut std::io::stdout()).expect("failed to flush stdout");
            }
        }
        let mut m = Self::zeros(shape_info[0], shape_info[1])?;
        let mut dcnt: usize = 0;
        let mut mdata: Vec<Result<_, _>>;
        println!("matrix data:");
        while dcnt < m.data.len() {
            rb.clear();
            if std::io::stdin().read_line(&mut rb).is_ok() && {
                mdata = rb
                    .split(|c: char| c.is_whitespace())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.trim().parse::<N>())
                    .collect();
                mdata
                    .iter()
                    .map(|v| v.is_ok())
                    .fold(true, |acc, v| acc && v)
            } {
                mdata.iter().for_each(|v| {
                    if dcnt < m.data.len() {
                        if let Ok(v) = v {
                            m.data[dcnt] = v.clone();
                            dcnt += 1;
                        }
                    }
                });
            } else {
                eprintln!("failed to get matrix data, please re-input!");
                dcnt = 0;
                println!("matrix data:");
            }
        }
        Ok(m)
    }

    pub fn p_change(n: usize, i: usize, j: usize) -> Result<Matrix<N>, RMatrixError> {
        let mut m = Self::eyes(n, n)?;
        m.set(N::default(), i, i)?;
        m.set(N::default(), j, j)?;
        m.set(N::one(), i, j)?;
        m.set(N::one(), j, i)?;
        Ok(m)
    }

    pub fn p_add(n: usize, k: N, i: usize, j: usize) -> Result<Self, RMatrixError> {
        //! add `k` times row/column `i` to row/column `i`
        //!
        //! # Examples
        //!
        //! ```rust
        //! # use rmatrix::matrix::Matrix;
        //! let a = Matrix::<f64>::eyes(2, 2).unwrap();
        //! // add 2.0 times row 2 to row 1
        //! let pa = a.times(&Matrix::<f64>::p_add(2, 2.0, 2, 1).unwrap()).unwrap();
        //! assert_eq!(pa, Matrix::from_vec(2, 2, vec![1.0, 2.0, 0.0, 1.0]).unwrap());
        //! ```

        let mut m = Self::eyes(n, n)?;
        m.set(k, j, i)?;
        Ok(m)
    }

    pub fn p_smul(n: usize, k: N, i: usize) -> Result<Self, RMatrixError> {
        let mut m = Self::eyes(n, n)?;
        m.set(k, i, i)?;
        Ok(m)
    }

    pub fn dot(v1: Vec<N>, v2: Vec<N>) -> Result<N, RMatrixError> {
        if v1.len() == v2.len() {
            Ok(std::iter::zip(v1, v2).map(|(e1, e2)| e1 * e2).sum())
        } else {
            Err(RMatrixError::LengthInconsistent(v1.len(), v2.len()))
        }
    }

    pub fn outer(v1: Vec<N>, v2: Vec<N>) -> Result<Self, RMatrixError> {
        let mut m = Self::zeros(v1.len(), v2.len())?;
        for i in 0..v1.len() {
            for j in 0..v2.len() {
                m.set(v1[i] * v2[j], i + 1, j + 1)?;
            }
        }
        Ok(m)
    }
}

impl Matrix<f64> {
    pub fn rand(
        r: usize,
        c: usize,
        lb: f64,
        ub: f64,
    ) -> Result<Matrix<f64>, Box<dyn std::error::Error>> {
        let mut m = Self::zeros(r, c)?;
        for i in 0..m.data.len() {
            m.data[i] = thread_rng().gen_range(lb.min(ub)..ub.max(lb));
        }
        Ok(m)
    }
}

impl<N: Number> std::fmt::Display for Matrix<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "")?;
        for r in 1..=self.shape.row {
            write!(f, "[")?;
            for c in 1..=self.shape.col - 1 {
                write!(f, "{:>6.3}, ", self.get(r, c).unwrap())?;
            }
            writeln!(f, "{:>6.3}]", self.get(r, self.shape.col).unwrap())?;
        }
        writeln!(
            f,
            "<mat[{}] {}x{}>",
            self.tag, self.shape.row, self.shape.col
        )
    }
}

impl<N: Number> PartialEq for Matrix<N> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}
