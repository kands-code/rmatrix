use crate::matrix::shape::MatrixShape;
use crate::matrix::Matrix;
use rand::{distributions::Alphanumeric, thread_rng, Rng};
use std::io::Write;

impl Matrix {
    pub fn zeros(r: usize, c: usize) -> Result<Self, String> {
        //! return a zero matrix with specific size
        //!
        //! # Examples
        //!
        //! ```rust
        //! # use rmatrix::matrix::Matrix;
        //! // get a 3x3 zero matrix
        //! let zero_matrix = Matrix::zeros(3, 3);
        //! ```

        fn random_string(n: usize) -> String {
            thread_rng()
                .sample_iter(&Alphanumeric)
                .take(n)
                .map(char::from)
                .collect()
        }

        Ok(Matrix {
            shape: MatrixShape::new(r, c)?,
            data: vec![0.0; r * c],
            tag: random_string(8),
        })
    }

    pub fn eyes(r: usize, c: usize) -> Result<Matrix, String> {
        let mut m = Self::zeros(r, c)?;
        for i in 1..=r.min(c) {
            m.set(1.0, i, i)?;
        }
        Ok(m)
    }

    pub fn rand(
        r: usize,
        c: usize,
        lb: f64,
        ub: f64,
    ) -> Result<Matrix, Box<dyn std::error::Error>> {
        let mut m = Self::zeros(r, c)?;
        for i in 0..m.data.len() {
            m.data[i] = thread_rng().gen_range(lb.min(ub)..ub.max(lb));
        }
        Ok(m)
    }

    pub fn from_vec(row: usize, col: usize, data: Vec<f64>) -> Result<Self, String> {
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

    pub fn from_stdin() -> Result<Self, String> {
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
        //! ```rust,no_run
        //! # use rmatrix::matrix::Matrix;
        //! let m = Matrix::from_stdin().unwrap();
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

        print!("matrix shape (r, c): ");
        std::io::Write::flush(&mut std::io::stdout()).expect("failed to flush stdout");
        let mut rb: String = String::new();
        let mut shape_info: Vec<usize>;
        loop {
            rb.clear();
            if std::io::stdin().read_line(&mut rb).is_ok() {
                shape_info = rb
                    .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<usize>().unwrap_or_default())
                    .collect::<Vec<_>>();
            } else {
                loop {
                    eprintln!("failed to get matrix shape info, please re-input!");
                    if let Ok(_) = std::io::stdin().read_line(&mut rb) {
                        shape_info = rb
                            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
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
                    .map(|s| s.trim().parse::<f64>())
                    .collect();
                mdata
                    .iter()
                    .map(|v| v.is_ok())
                    .fold(true, |acc, v| acc && v)
            } {
                mdata.iter().for_each(|v| {
                    if dcnt < m.data.len() {
                        m.data[dcnt] = v.clone().unwrap();
                        dcnt += 1;
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

    pub fn from_json<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        Ok(serde_json::from_reader(std::io::BufReader::new(
            std::fs::File::open(path)?,
        ))?)
    }

    pub fn to_json<P: AsRef<std::path::Path>>(
        data: &Vec<Self>,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        write!(
            std::fs::File::create(path)?,
            "{}",
            serde_json::to_string(data)?
        )?;
        Ok(())
    }

    pub fn p_change(n: usize, i: usize, j: usize) -> Result<Matrix, String> {
        let mut m = Self::eyes(n, n)?;
        m.set(0.0, i, i)?;
        m.set(0.0, j, j)?;
        m.set(1.0, i, j)?;
        m.set(1.0, j, i)?;
        Ok(m)
    }

    pub fn p_add(n: usize, k: f64, i: usize, j: usize) -> Result<Self, String> {
        //! add `k` times row/column `i` to row/column `i`
        //!
        //! # Examples
        //!
        //! ```rust
        //! # use rmatrix::matrix::Matrix;
        //! let a = Matrix::eyes(2, 2).unwrap();
        //! // add 2.0 times row 2 to row 1
        //! let pa = a.mul(&Matrix::p_add(2, 2.0, 2, 1).unwrap()).unwrap();
        //! assert_eq!(pa, Matrix::from_vec(2, 2, vec![1.0, 2.0, 0.0, 1.0]).unwrap());
        //! ```

        let mut m = Self::eyes(n, n)?;
        m.set(k, j, i)?;
        Ok(m)
    }

    pub fn p_smul(n: usize, k: f64, i: usize) -> Result<Self, String> {
        let mut m = Self::eyes(n, n)?;
        m.set(k, i, i)?;
        Ok(m)
    }

    pub fn inner(v1: Vec<f64>, v2: Vec<f64>) -> Result<f64, String> {
        if v1.len() == v2.len() {
            Ok(std::iter::zip(v1, v2).map(|(e1, e2)| e1 * e2).sum())
        } else {
            Err(format!("vectors must have same length"))
        }
    }

    pub fn outter(v1: Vec<f64>, v2: Vec<f64>) -> Result<Self, String> {
        let mut m = Self::zeros(v1.len(), v2.len())?;
        for i in 0..v1.len() {
            for j in 0..v2.len() {
                m.set(v1[i] * v2[j], i + 1, j + 1)?;
            }
        }
        Ok(m)
    }
}

impl std::fmt::Display for Matrix {
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

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}
