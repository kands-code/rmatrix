pub mod complex;
pub mod file;
pub mod number;
pub mod vector;

use serde::Deserialize;

use crate::number::INum;
use crate::vector::Vector;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct Matrix<N, const R: usize, const C: usize> {
    data: Vec<N>,
}

impl<'de, N: INum<'de>, const R: usize, const C: usize> Matrix<N, R, C> {
    /// constructor of matrices
    ///
    /// ```rust,no_run
    /// # use rmatrix_ks::Matrix;
    /// # fn main() {
    /// let _: Matrix<i32, 2, 2> = Matrix::new(&[1, 2, 3, 4]);
    /// # }
    /// ```
    ///
    /// if data length is not long enough,
    /// the constructor will panic
    ///
    /// ```rust,should_panic
    /// # use rmatrix_ks::Matrix;
    /// # fn main() {
    /// let _: Matrix<i32, 3, 2> = Matrix::new(&[1, 2, 3, 4]);
    /// # }
    /// ```
    pub fn new(data: &[N]) -> Self {
        assert!(R > 0 && C > 0);
        assert!(data.len() >= R * C);
        let data = data.split_at(R * C).0;
        Matrix {
            data: data.to_vec(),
        }
    }

    /// another constructor of matrices, get a ZERO matrix of size (r, c)
    ///
    /// ```rust,no_run
    /// # use rmatrix_ks::Matrix;
    /// # fn main() {
    /// let _: Matrix<i32, 3, 3> = Matrix::zeros();
    /// # }
    /// ```
    pub fn zeros() -> Self {
        assert!(R > 0 && C > 0);
        Matrix {
            data: vec![N::default(); R * C],
        }
    }

    /// another constructor of matrices, get a diag matrix of size (r, c)
    ///
    /// ```rust,no_run
    /// # use rmatrix_ks::Matrix;
    /// # fn main() {
    /// let _: Matrix<i32, 3, 3> = Matrix::diag(&[1, 2, 3]);
    /// # }
    /// ```
    pub fn diag(data: &[N]) -> Self {
        let min_side = R.min(C);
        assert!(data.len() >= min_side);
        let mut diag_matrix = Matrix::<N, R, C>::zeros();
        for i in 1..=min_side {
            diag_matrix.set(i, i, data.get(i - 1).unwrap().clone());
        }
        diag_matrix
    }

    /// another constructor of matrices, get an identity matrix of size (r, c)
    ///
    /// ```rust,no_run
    /// # use rmatrix_ks::Matrix;
    /// # fn main() {
    /// let _: Matrix<i32, 3, 3> = Matrix::ident();
    /// # }
    /// ```
    pub fn ident() -> Self {
        Matrix::<N, R, C>::diag(&vec![N::one(); R.min(C)])
    }

    /// the Getter of matrices
    ///
    /// ```rust
    /// # use rmatrix_ks::Matrix;
    /// # fn main() {
    /// let m = Matrix::<_, 3, 2>::new(&[1, 2, 3, 4, 5, 6]);
    /// assert_eq!(m.get(2, 1), &3);
    /// # }
    /// ```
    pub fn get(&self, rth: usize, cth: usize) -> &N {
        assert!(rth > 0 && rth <= R && cth > 0 && cth <= C);
        self.data.get((rth - 1) * C + cth - 1).unwrap()
    }

    /// the Setter of matrices
    ///
    /// ```rust
    /// # use rmatrix_ks::Matrix;
    /// # fn main() {
    /// let mut m = Matrix::<_,3, 2>::new(&[1, 2, 3, 4, 5, 6]);
    /// m.set(2, 1, 10);
    /// assert_eq!(m.get(2, 1), &10);
    /// # }
    /// ```
    pub fn set(&mut self, rth: usize, cth: usize, val: N) {
        assert!(rth > 0 && rth <= R && cth > 0 && cth <= C);
        self.data[(rth - 1) * C + cth - 1] = val;
    }

    /// get the row vector of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::Matrix;
    /// # fn main() {
    /// let m = Matrix::<_, 3, 2>::new(&[1, 2, 3, 4, 5, 6]);
    /// assert_eq!(m.get_nth_row(2), Matrix::<_, 1, 2>::new(&[3, 4]));
    /// # }
    /// ```
    pub fn get_nth_row(&self, nth: usize) -> Matrix<N, 1, C> {
        assert!(nth > 0 && nth <= R);
        let mut nth_row = Matrix::<N, 1, C>::zeros();
        for i in 1..=C {
            nth_row.set(1, i, self.get(nth, i).clone())
        }
        nth_row
    }

    /// get the column vector of the matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::Matrix;
    /// # fn main() {
    /// let m = Matrix::<_, 3, 2>::new(&[1, 2, 3, 4, 5, 6]);
    /// assert_eq!(m.get_nth_col(2), Matrix::<_, 3, 1>::new(&[2, 4, 6]));
    /// # }
    /// ```
    pub fn get_nth_col(&self, nth: usize) -> Vector<N, R> {
        assert!(nth > 0 && nth <= C);
        let mut nth_col = Matrix::<N, R, 1>::zeros();
        for i in 1..=R {
            nth_col.set(i, 1, self.get(i, nth).clone())
        }
        nth_col
    }

    /// checking if a matrix is a ZERO matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::Matrix;
    /// # fn main() {
    /// let m = Matrix::<_, 3, 2>::zeros();
    /// assert!(m.is_zero());
    ///
    /// let n = Matrix::<_, 3, 2>::new(&[1, 2, 3, 4, 5, 6]);
    /// assert!(!n.is_zero());
    /// # }
    /// ```
    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|x| x.is_zero())
    }

    /// transpose a matrix
    ///
    /// ```rust
    /// # use rmatrix_ks::Matrix;
    /// # fn main() {
    /// let m1 = Matrix::<_, 3, 2>::new(&[1, 2, 3, 4, 5, 6]);
    /// let m2 = Matrix::<_, 2, 3>::new(&[1, 3, 5, 2, 4, 6]);
    /// assert_eq!(m1.transpose(), m2);
    /// # }
    /// ```
    pub fn transpose(&self) -> Matrix<N, C, R> {
        let mut tm = Matrix::zeros();
        for i in 1..=R {
            for j in 1..=C {
                tm.set(j, i, self.get(i, j).clone());
            }
        }
        tm
    }

    pub fn smap(&self, f: impl Fn(N) -> N) -> Self {
        let mut sm = Matrix::zeros();
        for i in 1..=R {
            for j in 1..=C {
                sm.set(i, j, f(self.get(i, j).clone()))
            }
        }
        sm
    }

    pub fn sadd(&self, v: N) -> Self {
        self.smap(|x| x + v.clone())
    }

    pub fn smul(&self, v: N) -> Self {
        self.smap(|x| x * v.clone())
    }
}

impl<'de, N: INum<'de>, const R: usize, const C: usize> std::fmt::Display for Matrix<N, R, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "\n<<matrix>>")?;
        for i in 1..=R {
            for j in 1..=C {
                write!(f, " {} ", self.get(i, j))?
            }
            writeln!(f)?
        }
        writeln!(f, "<<{}x{}>>", R, C)
    }
}
