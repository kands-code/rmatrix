//! # matrix::utils
//!
//! Some util functions.

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::vector::is_row_vector;
use crate::{
    matrix::{
        Matrix,
        math::{is_square_matrix, row_eliminate, row_reduce},
        vector::{is_column_vector, layer_product, normalize, project_to},
    },
    number::traits::{fractional::Fractional, number::Number, realfloat::RealFloat},
};

/// Generates coordinates within a specified inclusive-range that meet certain criteria.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::matrix::utils::points_2d;
///
/// // Generates coordinates for the lower triangular part of a 3x3 matrix.
/// let p = points_2d((1, 3), (1, 3), |row, column| row > column);
/// assert_eq!(p, vec![(2, 1), (3, 1), (3, 2)]);
/// ```
pub fn points_2d<F>((row_lb, row_ub): (usize, usize), (col_lb, col_ub): (usize, usize), criteria: F) -> Vec<(usize, usize)>
where
    F: Fn(usize, usize) -> bool,
{
    if row_lb > row_ub || col_lb > col_ub {
        Vec::new()
    } else {
        let mut all_points = Vec::with_capacity((row_ub - row_lb + 1) * (col_ub - col_lb + 1));
        for row in row_lb..=row_ub {
            for col in col_lb..=col_ub {
                if criteria(row, col) {
                    all_points.push((row, col));
                }
            }
        }
        // used to save a certain amount of space
        all_points.shrink_to_fit();
        all_points
    }
}

/// Calculates the trace of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, utils::trace},
///     number::instances::word8::Word8,
/// };
///
/// let m =
///     Matrix::<Word8>::of(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
/// assert_eq!(trace(&m), Word8::of(15));
/// ```
pub fn trace<N>(m: &Matrix<N>) -> N
where
    N: Number,
{
    m.get_diagonal()
        .linear_iter()
        .cloned()
        .fold(N::zero(), |acc, e| acc + e.clone())
}

/// Applies the function f to each element of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, utils::apply},
///     number::instances::word8::Word8,
/// };
///
/// let m =
///     Matrix::<Word8>::of(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
/// let n = Matrix::<Word8>::of(
///     3,
///     3,
///     &[2, 4, 6, 8, 10, 12, 14, 16, 18].map(|e| Word8::of(e)),
/// )
/// .unwrap();
/// assert_eq!(apply(&m, |e| e * Word8::of(2)), n);
/// ```
pub fn apply<N, M, F>(m: &Matrix<N>, f: F) -> Matrix<M>
where
    N: Sync + Clone,
    M: Send + Sync,
    F: Fn(N) -> M + Sync,
{
    let inner = m.inner.par_iter().map(|e| f(e.clone())).collect();
    Matrix {
        inner,
        row: m.row,
        column: m.column,
    }
}

/// Obtains the transpose of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, utils::transpose},
///     number::instances::word8::Word8,
/// };
///
/// let m =
///     Matrix::<Word8>::of(3, 3, &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e))).unwrap();
/// let n =
///     Matrix::<Word8>::of(3, 3, &[1, 4, 7, 2, 5, 8, 3, 6, 9].map(|e| Word8::of(e))).unwrap();
/// assert_eq!(transpose(&m), n);
/// ```
pub fn transpose<N>(m: &Matrix<N>) -> Matrix<N>
where
    N: Clone,
{
    let mut inner = Vec::with_capacity(m.column * m.row);
    for row_index in 1..=m.column {
        for column_index in 1..=m.row {
            inner.push(m[(column_index, row_index)].clone());
        }
    }
    Matrix {
        inner,
        row: m.column,
        column: m.row,
    }
}

/// Construct a projection matrix that can project all vectors onto the corresponding vector.
///
/// # Panics
///
/// The `to` must be a column vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, utils::project_matrix, vector::column_vector},
///     number::instances::float::Float,
/// };
///
/// let v1 = column_vector::<Float>(3, &[2.0, 3.0, 4.0].map(Float::of)).unwrap();
/// let p = project_matrix(&v1);
/// let p_expect = Matrix::<Float>::of(
///     3,
///     3,
///     &[4.0, 6.0, 8.0, 6.0, 9.0, 12.0, 8.0, 12.0, 16.0].map(|e| Float::of(e / 29.0)),
/// )
/// .unwrap();
/// assert_eq!(p, p_expect);
/// ```
pub fn project_matrix<N>(to: &Matrix<N>) -> Matrix<N>
where
    N: RealFloat,
{
    assert!(
        is_column_vector(to),
        concat!(
            "Error[matrix::utils::project_matrix]: ",
            "A column vector is required."
        )
    );

    let normalized = normalize(to);
    layer_product(&normalized, &transpose(&normalized))
}

/// Calculate the rank of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, utils::rank},
///     number::instances::double::Double,
/// };
///
/// let m = Matrix::<Double>::of(
///     3,
///     3,
///     &[2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0].map(|e| Double::of(e)),
/// )
/// .unwrap();
/// assert_eq!(rank(&m), 3);
/// ```
pub fn rank<N>(m: &Matrix<N>) -> usize
where
    N: Fractional,
{
    let (_, _, _, reduced) = row_reduce(m);
    (1..=m.row)
        .map(|row_index| {
            reduced
                .get_row(row_index)
                .expect(&format!(
                    concat!(
                        "Error[matrix::utils::rank]: ",
                        "Failed to retrieve the {}-th row of the matrix."
                    ),
                    row_index
                ))
                .inner
                .par_iter()
                .any(|e| !e.is_zero())
        })
        .filter(|&p| p)
        .count()
}

/// Calculate the null space of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, utils::null_space, vector::column_vector},
///     number::instances::double::Double,
/// };
///
/// let a = Matrix::<Double>::of(
///     3,
///     5,
///     &[
///         -3.0, 6.0, -1.0, 1.0, -7.0, 1.0, -2.0, 2.0, 3.0, -1.0, 2.0, -4.0, 5.0, 8.0, -4.0,
///     ]
///     .map(Double::of),
/// )
/// .unwrap();
/// let ns = null_space(&a);
/// // For this matrix, the null space contains only three elements.
/// assert_eq!(ns.len(), 3);
/// // Second column:
/// let n1 = column_vector::<Double>(5, &[2.0, 1.0, 0.0, 0.0, 0.0].map(Double::of)).unwrap();
/// assert_eq!(ns[0], n1);
/// // Fourth column:
/// let n2 = column_vector::<Double>(5, &[1.0, 0.0, -2.0, 1.0, 0.0].map(Double::of)).unwrap();
/// assert_eq!(ns[1], n2);
/// // Fifth column:
/// let n3 = column_vector::<Double>(5, &[-3.0, 0.0, 2.0, 0.0, 1.0].map(Double::of)).unwrap();
/// assert_eq!(ns[2], n3);
/// ```
pub fn null_space<N>(m: &Matrix<N>) -> Vec<Matrix<N>>
where
    N: Fractional,
{
    // Reduce the matrix to its row echelon form.
    let refined = row_eliminate(m);
    // Set the markers for each row, which is the column index of the first non-zero element.
    let mut row_flags = vec![0; m.column];
    for row in 1..=m.row {
        for column in row..=m.column {
            if !refined[(row, column)].is_zero() {
                row_flags[column - 1] = row;
                break;
            }
        }
    }
    let mut space = Vec::new();
    // Only rank-deficient matrices have a nullspace.
    if row_flags.contains(&0) {
        for column in 1..=m.column {
            // The vector corresponding to unmarked columns is an element of the nullspace.
            if row_flags[column - 1] == 0 {
                let mut base = Matrix::<N>::defaults(m.column, 1);
                for check in 1..=m.column {
                    if row_flags[check - 1] != 0 {
                        // Negate the elements of the marked rows.
                        let val = refined[(row_flags[check - 1], column)].clone();
                        base[(check, 1)] = if val.is_zero() { val } else { -val };
                    } else if column == check {
                        // Set the column corresponding to itself to one.
                        base[(check, 1)] = N::one();
                    }
                }
                space.push(base);
            }
        }
    }
    space
}

/// Find the column space of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, utils::column_space, vector::column_vector},
///     number::instances::float::Float,
/// };
///
/// let a = Matrix::<Float>::of(
///     3,
///     5,
///     &[
///         -3.0, 6.0, -1.0, 1.0, -7.0, 1.0, -2.0, 2.0, 3.0, -1.0, 2.0, -4.0, 5.0, 8.0, -4.0,
///     ]
///     .map(Float::of),
/// )
/// .unwrap();
/// let column_space_a = column_space(&a);
/// // The column space contains only two elements.
/// assert_eq!(column_space_a.len(), 2);
/// assert_eq!(
///     column_space_a[0],
///     column_vector::<Float>(3, &[-3.0, 1.0, 2.0].map(Float::of)).unwrap()
/// );
/// assert_eq!(
///     column_space_a[1],
///     column_vector::<Float>(3, &[-1.0, 2.0, 5.0].map(Float::of)).unwrap()
/// );
/// ```
pub fn column_space<N>(m: &Matrix<N>) -> Vec<Matrix<N>>
where
    N: Fractional,
{
    // Reduce the matrix to its row echelon form.
    let refined = row_eliminate(m);
    // Set the markers for each row, which is the column index of the first non-zero element.
    let mut cols = Vec::new();
    for row in 1..=m.row {
        for column in row..=m.column {
            if !refined[(row, column)].is_zero() {
                cols.push(column);
                break;
            }
        }
    }
    cols.iter()
        .map(|&c| {
            apply(
                &m.get_column(c).expect(&format!(
                    concat!(
                        "Error[matrix::utils::column_space]: ",
                        "Failed to retrieve the {}-th column of the matrix."
                    ),
                    c,
                )),
                |e: &N| e.clone(),
            )
        })
        .collect::<Vec<Matrix<N>>>()
}

/// Find the row space of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, utils::row_space, vector::column_vector},
///     number::instances::float::Float,
/// };
///
/// let a = Matrix::<Float>::of(
///     3,
///     5,
///     &[
///         -3.0, 6.0, -1.0, 1.0, -7.0, 1.0, -2.0, 2.0, 3.0, -1.0, 2.0, -4.0, 5.0, 8.0, -4.0,
///     ]
///     .map(Float::of),
/// )
/// .unwrap();
/// let row_space_a = row_space(&a);
/// // The column space contains only two elements.
/// assert_eq!(row_space_a.len(), 2);
/// assert_eq!(
///     row_space_a[0],
///     column_vector::<Float>(5, &[-3.0, 6.0, -1.0, 1.0, -7.0].map(Float::of)).unwrap()
/// );
/// assert_eq!(
///     row_space_a[1],
///     column_vector::<Float>(5, &[1.0, -2.0, 2.0, 3.0, -1.0].map(Float::of)).unwrap()
/// );
/// ```
pub fn row_space<N>(m: &Matrix<N>) -> Vec<Matrix<N>>
where
    N: Fractional,
{
    column_space(&transpose(m))
}

/// Horizontally concatenate two matrices.
///
/// # Panics
///
/// The number of rows in `m1` and `m2` must be the same.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, utils::horizontal_concat},
///     number::instances::word8::Word8,
/// };
///
/// let m = Matrix::<Word8>::of(2, 2, &[1, 2, 3, 4].map(|e| Word8::of(e))).unwrap();
/// let n = Matrix::<Word8>::of(2, 2, &[5, 6, 7, 8].map(|e| Word8::of(e))).unwrap();
/// let cat =
///     Matrix::<Word8>::of(2, 4, &[1, 2, 5, 6, 3, 4, 7, 8].map(|e| Word8::of(e))).unwrap();
/// assert_eq!(horizontal_concat(&m, &n), cat);
/// ```
pub fn horizontal_concat<N>(m1: &Matrix<N>, m2: &Matrix<N>) -> Matrix<N>
where
    N: Clone,
{
    assert_eq!(
        m1.row, m2.row,
        concat!(
            "Error[matrix::utils::horizontal_concat]: ",
            "The number of rows in m1 and m2 must be the same."
        )
    );

    let mut inner = Vec::with_capacity(m1.row * (m1.column + m2.column));
    for row_index in 1..=m1.row {
        for column_index in 1..=(m1.column + m2.column) {
            inner.push(if column_index <= m1.column {
                m1[(row_index, column_index)].clone()
            } else {
                m2[(row_index, column_index - m1.column)].clone()
            });
        }
    }
    Matrix {
        inner,
        row: m2.row,
        column: m1.column + m2.column,
    }
}

/// Vertically concatenate two matrices.
///
/// # Panics
///
/// The number of columns in `m1` and `m2` must be the same.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, utils::vertical_concat},
///     number::instances::word8::Word8,
/// };
///
/// let m = Matrix::<Word8>::of(2, 2, &[1, 2, 3, 4].map(|e| Word8::of(e))).unwrap();
/// let n = Matrix::<Word8>::of(2, 2, &[5, 6, 7, 8].map(|e| Word8::of(e))).unwrap();
/// let cat =
///     Matrix::<Word8>::of(4, 2, &[1, 2, 3, 4, 5, 6, 7, 8].map(|e| Word8::of(e))).unwrap();
/// assert_eq!(vertical_concat(&m, &n), cat);
/// ```
pub fn vertical_concat<N>(m1: &Matrix<N>, m2: &Matrix<N>) -> Matrix<N>
where
    N: Clone,
{
    assert_eq!(
        m1.column, m2.column,
        concat!(
            "Error[matrix::utils::horizontal_concat]: ",
            "The number of columns in m1 and m2 must be the same."
        )
    );

    let mut inner = Vec::with_capacity((m1.row + m2.row) * m1.column);
    for row_index in 1..=(m1.row + m2.row) {
        for column_index in 1..=m1.column {
            inner.push(if row_index <= m1.row {
                m1[(row_index, column_index)].clone()
            } else {
                m2[(row_index - m1.row, column_index)].clone()
            });
        }
    }
    Matrix {
        inner,
        row: m1.row + m2.row,
        column: m2.column,
    }
}

/// Constructing a matrix from column vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, utils::from_columns, vector::column_vector},
///     number::instances::word8::Word8,
/// };
///
/// let v1 = column_vector(3, &[1, 2, 3].map(Word8::of)).unwrap();
/// let v2 = column_vector(3, &[4, 7, 9].map(Word8::of)).unwrap();
/// let v3 = column_vector(3, &[5, 8, 6].map(Word8::of)).unwrap();
/// let m = from_columns(3, 3, &[v1, v2, v3]);
/// let m_expect = Matrix::of(3, 3, &[1, 4, 5, 2, 7, 8, 3, 9, 6].map(Word8::of));
/// assert_eq!(m, m_expect);
/// ```
pub fn from_columns<N>(row: usize, column: usize, columns: &[Matrix<N>]) -> Option<Matrix<N>>
where
    N: Clone + Sync,
{
    if columns
        .par_iter()
        .all(|v| is_column_vector(v) && v.row == row)
    {
        let mut inner = Vec::with_capacity(row * column);
        for row_index in 1..=row {
            columns
                .iter()
                .take(column)
                .for_each(|e| inner.push(e[(row_index, 1)].clone()));
        }
        Some(Matrix { inner, row, column })
    } else {
        eprintln!(concat!("Error[matrix::utils::from_columns]: ", ""));
        None
    }
}

/// Constructing a matrix from row vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, utils::from_rows, vector::row_vector},
///     number::instances::word8::Word8,
/// };
///
/// let v1 = row_vector(3, &[1, 2, 3].map(Word8::of)).unwrap();
/// let v2 = row_vector(3, &[4, 7, 9].map(Word8::of)).unwrap();
/// let v3 = row_vector(3, &[5, 8, 6].map(Word8::of)).unwrap();
/// let m = from_rows(3, 3, &[v1, v2, v3]);
/// let m_expect = Matrix::of(3, 3, &[1, 2, 3, 4, 7, 9, 5, 8, 6].map(Word8::of));
/// assert_eq!(m, m_expect);
/// ```
pub fn from_rows<N>(row: usize, column: usize, rows: &[Matrix<N>]) -> Option<Matrix<N>>
where
    N: Clone + Sync,
{
    if rows
        .par_iter()
        .all(|v| is_row_vector(v) && v.column == column)
    {
        let mut inner = Vec::with_capacity(row * column);
        rows.iter().take(row).for_each(|e| {
            for column_index in 1..=column {
                inner.push(e[(1, column_index)].clone());
            }
        });
        Some(Matrix { inner, row, column })
    } else {
        eprintln!(concat!("Error[matrix::utils::from_rows]: ", ""));
        None
    }
}

/// Use the Gram-Schmidt process
/// to find the orthogonal basis corresponding to the given set of vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         Matrix,
///         utils::{apply, gram_schmidt_process},
///         vector::dot_product,
///     },
///     number::{
///         instances::float::Float,
///         traits::{one::One, zero::Zero},
///     },
/// };
///
/// let basis =
///     Matrix::<Float>::of(3, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 1.0].map(Float::of)).unwrap();
/// let ob = gram_schmidt_process(&basis);
/// let c1 = apply(&ob.get_column(1).unwrap(), |e: &Float| e.clone());
/// let c2 = apply(&ob.get_column(2).unwrap(), |e: &Float| e.clone());
/// // Each column vector is normalized.
/// assert_eq!(
///     (dot_product(&c1, &c1), dot_product(&c2, &c2)),
///     (Float::one(), Float::one())
/// );
/// // The column vectors are mutually orthogonal.
/// assert_eq!(dot_product(&c1, &c2), Float::zero());
/// // Corresponding orthogonal basis.
/// let ob_expect = Matrix::<Float>::of(
///     3,
///     2,
///     &[
///         1.0 / 3.0f32.sqrt(),
///         -2.0 / 6.0f32.sqrt(),
///         1.0 / 3.0f32.sqrt(),
///         1.0 / 6.0f32.sqrt(),
///         1.0 / 3.0f32.sqrt(),
///         1.0 / 6.0f32.sqrt(),
///     ]
///     .map(Float::of),
/// )
/// .unwrap();
/// assert_eq!(ob, ob_expect);
/// ```
pub fn gram_schmidt_process<N>(basis: &Matrix<N>) -> Matrix<N>
where
    N: RealFloat,
{
    let mut orthonormal_basis = Matrix::defaults(basis.row, basis.column);
    for k in 1..=basis.column {
        // uk1 = bk
        // where Basis = [b1 | b2 | ... | bc]
        let mut uk = apply(
            &basis.get_column(k).expect(concat!(
                "Error[matrix::utils::gram_schmidt_process]: ",
                "Failed to retrieve the column vector of basis."
            )),
            |e: &N| e.clone(),
        );
        // uk = bk - sum(proj(bk, uj), (j, 1, k - 1))
        for j in 1..k {
            // Use MGS, ukj = uk(j - 1) - proj(uk(j - 1), uj)
            let uj = apply(
                &orthonormal_basis.get_column(j).expect(concat!(
                    "Error[matrix::utils::gram_schmidt_process]: ",
                    "Failed to retrieve the column vector of orthonormal_basis."
                )),
                |e: &N| e.clone(),
            );
            uk = uk.clone() - project_to(&uk, &uj);
        }
        // Normalize uk.
        uk = normalize(&uk);
        for row in 1..=basis.row {
            // OB = [u1 | u2 | ... | uc]
            orthonormal_basis[(row, k)] = uk[(row, 1)].clone();
        }
    }
    orthonormal_basis
}

/// Generate the corresponding real Givens rotation matrix
/// to eliminate the element at '(row, column)' using '(row - 1, column)'.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{Matrix, math::is_orthogonal_matrix, utils::givens_rotation_matrix},
///     number::instances::float::Float,
/// };
///
/// let m = Matrix::<Float>::of(
///     4,
///     4,
///     &[
///         1.0, 2.9, 3.8, 4.7, 5.6, 6.5, 7.4, 8.3, 9.2, 10.1, 11.0, 12.9, 13.8, 14.7, 15.6,
///         16.5,
///     ]
///     .map(Float::of),
/// )
/// .unwrap();
/// let g = givens_rotation_matrix(&m, 4, 1);
/// assert!(is_orthogonal_matrix(&g));
/// let p = g * m;
/// let p_expect = Matrix::<Float>::of(
///     4,
///     4,
///     &[
///         1.0, 2.9, 3.8, 4.7, 5.6, 6.5, 7.4, 8.3, 16.5855, 17.8336, 19.0817, 20.8845, 0.0,
///         -0.2496, -0.4992, -1.5809,
///     ]
///     .map(Float::of),
/// )
/// .unwrap();
/// assert_eq!(p, p_expect);
/// ```
pub fn givens_rotation_matrix<N>(m: &Matrix<N>, row: usize, column: usize) -> Matrix<N>
where
    N: RealFloat,
{
    let mut givens = Matrix::<N>::eyes(m.row, m.row);
    // e1 is the element to be eliminated.
    let e1 = m[(row, column)].clone();
    // e2 is the element used for elimination.
    let e2 = m[(row - 1, column)].clone();
    // v = (e2, e1), r = ||v||
    let r = (e1.clone() * e1.clone() + e2.clone() * e2.clone()).square_root();
    // sin(theta) = y / r = e1 / r
    let sin_theta = e1 / r.clone();
    // cos(theta) = x / r = e2 / r
    let cos_theta = e2 / r;
    // rotation {{c, s}, {-s, c}}
    givens[(row, row)] = cos_theta.clone();
    givens[(row - 1, row - 1)] = cos_theta;
    givens[(row, row - 1)] = -sin_theta.clone();
    givens[(row - 1, row)] = sin_theta;
    givens
}

/// Decompose the real matrix into
/// an orthogonal matrix and the corresponding Hessenberg matrix.
///
/// # Panics
///
/// Only square matrices can undergo the Heisenberg decomposition.
///
/// # Examples
///
/// ```rust
/// use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
/// use rmatrix_ks::{
///     matrix::{
///         Matrix,
///         math::is_orthogonal_matrix,
///         utils::{hessenberg_decomposition, points_2d, transpose},
///     },
///     number::{instances::float::Float, traits::zero::Zero},
/// };
///
/// let m = Matrix::<Float>::of(
///     4,
///     4,
///     &[
///         1.0, 2.9, 3.8, 4.7, 5.6, 6.5, 7.4, 8.3, 9.2, 10.1, 11.0, 12.9, 13.8, 14.7, 15.6,
///         16.5,
///     ]
///     .map(Float::of),
/// )
/// .unwrap();
/// let (p, h) = hessenberg_decomposition(&m);
/// assert!(
///     points_2d((1, 4), (1, 4), |r, c| r > c + 1)
///         .par_iter()
///         .all(|&pi| h[pi].is_zero())
/// );
/// assert!(is_orthogonal_matrix(&p));
/// // P . H . P^T == M
/// assert!(p.clone() * h * transpose(&p) == m);
/// ```
pub fn hessenberg_decomposition<N>(m: &Matrix<N>) -> (Matrix<N>, Matrix<N>)
where
    N: RealFloat,
{
    assert!(
        is_square_matrix(m),
        concat!(
            "Error[matrix::utils::hessenberg_decomposition]: ",
            "Only square matrices can undergo the Heisenberg decomposition."
        )
    );

    if m.row < 3 {
        (Matrix::eyes(m.row, m.column), m.clone())
    } else {
        let mut hessen = m.clone();
        let mut p = Matrix::eyes(m.row, m.column);
        for column in 1..=m.column {
            for row in (column + 1..m.edge()).rev() {
                if !hessen[(row + 1, column)].is_zero() {
                    let pi = transpose(&givens_rotation_matrix(&hessen, row + 1, column));
                    hessen = transpose(&pi) * hessen * pi.clone();
                    p = p * pi;
                }
            }
        }
        (p, hessen)
    }
}
