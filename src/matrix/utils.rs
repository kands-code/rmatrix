//! # matrix::utils
//!
//! Some util functions.

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    matrix::{
        math::{row_eliminate, row_reduce},
        matrix::Matrix,
        vector::{VectorC, layer_product, normalize, project_to},
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
/// fn main() {
///     // Generates coordinates for the lower triangular part of a 3x3 matrix.
///     let p = points_2d((1, 3), (1, 3), |row, column| row > column);
///     assert_eq!(p, vec![(2, 1), (3, 1), (3, 2)]);
/// }
/// ```
pub fn points_2d<F>(
    (row_lb, row_ub): (usize, usize),
    (col_lb, col_ub): (usize, usize),
    criteria: F,
) -> Vec<(usize, usize)>
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
/// # Panics
///
/// This function requires the use of the `#![feature(generic_const_exprs)]`.
///
/// # Examples
///
/// ```rust
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::trace},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m = Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e)))
///         .unwrap();
///     assert_eq!(trace(&m), Word8::of(15));
/// }
/// ```
pub fn trace<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> N
where
    N: Number,
    [(); Matrix::<N, R, C>::get_diagonal_length()]:,
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
///     matrix::{matrix::Matrix, utils::apply},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m = Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e)))
///         .unwrap();
///     let n =
///         Matrix::<Word8, 3, 3>::of(&[2, 4, 6, 8, 10, 12, 14, 16, 18].map(|e| Word8::of(e)))
///             .unwrap();
///     assert_eq!(apply(&m, |e| e * Word8::of(2)), n);
/// }
/// ```
pub fn apply<N, M, F, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
    f: F,
) -> Matrix<M, R, C>
where
    N: Sync + Clone,
    M: Send + Sync,
    F: Fn(N) -> M + Sync,
{
    let inner = m.inner.par_iter().map(|e| f(e.clone())).collect();
    Matrix { inner }
}

/// Obtains the transpose of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::transpose},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m = Matrix::<Word8, 3, 3>::of(&[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|e| Word8::of(e)))
///         .unwrap();
///     let n = Matrix::<Word8, 3, 3>::of(&[1, 4, 7, 2, 5, 8, 3, 6, 9].map(|e| Word8::of(e)))
///         .unwrap();
///     assert_eq!(transpose(&m), n);
/// }
/// ```
pub fn transpose<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> Matrix<N, C, R>
where
    N: Clone,
{
    let mut inner = Vec::with_capacity(R * C);
    for row_index in 1..=C {
        for column_index in 1..=R {
            inner.push(m[(column_index, row_index)].clone());
        }
    }
    Matrix { inner }
}

/// Construct a projection matrix that can project all vectors onto the corresponding vector.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::project_matrix, vector::VectorC},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let v1 = VectorC::<Float, 3>::of(&[2.0, 3.0, 4.0].map(Float::of)).unwrap();
///     let p = project_matrix(&v1);
///     let p_expect = Matrix::<Float, 3, 3>::of(
///         &[4.0, 6.0, 8.0, 6.0, 9.0, 12.0, 8.0, 12.0, 16.0].map(|e| Float::of(e / 29.0)),
///     )
///     .unwrap();
///     assert_eq!(p, p_expect);
/// }
/// ```
pub fn project_matrix<N, const R: usize>(to: &VectorC<N, R>) -> Matrix<N, R, R>
where
    N: RealFloat,
{
    let normalized = normalize(to);
    layer_product(&normalized, &transpose(&normalized))
}

/// Calculate the rank of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::rank},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let m = Matrix::<Double, 3, 3>::of(
///         &[2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0].map(|e| Double::of(e)),
///     )
///     .unwrap();
///     assert_eq!(rank(&m), 3);
/// }
/// ```
pub fn rank<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> usize
where
    N: Fractional,
{
    let (_, _, _, reduced) = row_reduce(m);
    (1..=R)
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
///     matrix::{matrix::Matrix, utils::null_space, vector::VectorC},
///     number::instances::double::Double,
/// };
///
/// fn main() {
///     let a = Matrix::<Double, 3, 5>::of(
///         &[
///             -3.0, 6.0, -1.0, 1.0, -7.0, 1.0, -2.0, 2.0, 3.0, -1.0, 2.0, -4.0, 5.0, 8.0,
///             -4.0,
///         ]
///         .map(Double::of),
///     )
///     .unwrap();
///     let ns = null_space(&a);
///     // For this matrix, the null space contains only three elements.
///     assert_eq!(ns.len(), 3);
///     // Second column:
///     let n1 = VectorC::<Double, 5>::of(&[2.0, 1.0, 0.0, 0.0, 0.0].map(Double::of)).unwrap();
///     assert_eq!(ns[0], n1);
///     // Fourth column:
///     let n2 = VectorC::<Double, 5>::of(&[1.0, 0.0, -2.0, 1.0, 0.0].map(Double::of)).unwrap();
///     assert_eq!(ns[1], n2);
///     // Fifth column:
///     let n3 = VectorC::<Double, 5>::of(&[-3.0, 0.0, 2.0, 0.0, 1.0].map(Double::of)).unwrap();
///     assert_eq!(ns[2], n3);
/// }
/// ```
pub fn null_space<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> Vec<VectorC<N, C>>
where
    N: Fractional,
{
    // Reduce the matrix to its row echelon form.
    let refined = row_eliminate(m);
    // Set the markers for each row, which is the column index of the first non-zero element.
    let mut row_flags: [usize; C] = [0; C];
    for row in 1..=R {
        for column in row..=C {
            if !refined[(row, column)].is_zero() {
                row_flags[column - 1] = row;
                break;
            }
        }
    }
    let mut space = Vec::new();
    // Only rank-deficient matrices have a nullspace.
    if row_flags.iter().any(|&e| e == 0) {
        for column in 1..=C {
            // The vector corresponding to unmarked columns is an element of the nullspace.
            if row_flags[column - 1] == 0 {
                let mut base = VectorC::<N, C>::default();
                for check in 1..=C {
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
///     matrix::{matrix::Matrix, utils::column_space, vector::VectorC},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let a = Matrix::<Float, 3, 5>::of(
///         &[
///             -3.0, 6.0, -1.0, 1.0, -7.0, 1.0, -2.0, 2.0, 3.0, -1.0, 2.0, -4.0, 5.0, 8.0,
///             -4.0,
///         ]
///         .map(Float::of),
///     )
///     .unwrap();
///     let column_space_a = column_space(&a);
///     // The column space contains only two elements.
///     assert_eq!(column_space_a.len(), 2);
///     assert_eq!(
///         column_space_a[0],
///         VectorC::<Float, 3>::of(&[-3.0, 1.0, 2.0].map(Float::of)).unwrap()
///     );
///     assert_eq!(
///         column_space_a[1],
///         VectorC::<Float, 3>::of(&[-1.0, 2.0, 5.0].map(Float::of)).unwrap()
///     );
/// }
/// ```
pub fn column_space<N, const R: usize, const C: usize>(
    m: &Matrix<N, R, C>,
) -> Vec<VectorC<N, R>>
where
    N: Fractional,
{
    // Reduce the matrix to its row echelon form.
    let refined = row_eliminate(m);
    // Set the markers for each row, which is the column index of the first non-zero element.
    let mut cols = Vec::new();
    for row in 1..=R {
        for column in row..=C {
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
        .collect::<Vec<VectorC<N, R>>>()
}

/// Find the row space of the matrix.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::row_space, vector::VectorC},
///     number::instances::float::Float,
/// };
///
/// fn main() {
///     let a = Matrix::<Float, 3, 5>::of(
///         &[
///             -3.0, 6.0, -1.0, 1.0, -7.0, 1.0, -2.0, 2.0, 3.0, -1.0, 2.0, -4.0, 5.0, 8.0,
///             -4.0,
///         ]
///         .map(Float::of),
///     )
///     .unwrap();
///     let row_space_a = row_space(&a);
///     // The column space contains only two elements.
///     assert_eq!(row_space_a.len(), 2);
///     assert_eq!(
///         row_space_a[0],
///         VectorC::<Float, 5>::of(&[-3.0, 6.0, -1.0, 1.0, -7.0].map(Float::of)).unwrap()
///     );
///     assert_eq!(
///         row_space_a[1],
///         VectorC::<Float, 5>::of(&[1.0, -2.0, 2.0, 3.0, -1.0].map(Float::of)).unwrap()
///     );
/// }
/// ```
pub fn row_space<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> Vec<VectorC<N, C>>
where
    N: Fractional,
{
    column_space(&transpose(m))
}

/// Horizontally concatenate two matrices.
///
/// # Panics
///
/// This function requires the use of the `#![feature(generic_const_exprs)]`.
///
/// # Examples
///
/// ```rust
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::horizontal_concat},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m = Matrix::<Word8, 2, 2>::of(&[1, 2, 3, 4].map(|e| Word8::of(e))).unwrap();
///     let n = Matrix::<Word8, 2, 2>::of(&[5, 6, 7, 8].map(|e| Word8::of(e))).unwrap();
///     let cat =
///         Matrix::<Word8, 2, 4>::of(&[1, 2, 5, 6, 3, 4, 7, 8].map(|e| Word8::of(e))).unwrap();
///     assert_eq!(horizontal_concat(&m, &n), cat);
/// }
/// ```
pub fn horizontal_concat<N, const R: usize, const C1: usize, const C2: usize>(
    m1: &Matrix<N, R, C1>,
    m2: &Matrix<N, R, C2>,
) -> Matrix<N, R, { C1 + C2 }>
where
    N: Clone,
{
    let mut inner = Vec::with_capacity(R * (C1 + C2));
    for row_index in 1..=R {
        for column_index in 1..=(C1 + C2) {
            inner.push(if column_index <= C1 {
                m1[(row_index, column_index)].clone()
            } else {
                m2[(row_index, column_index - C1)].clone()
            });
        }
    }
    Matrix { inner }
}

/// Vertically concatenate two matrices.
///
/// # Panics
///
/// This function requires the use of the `#![feature(generic_const_exprs)]`.
///
/// # Examples
///
/// ```rust
/// #![allow(incomplete_features)]
/// #![feature(generic_const_exprs)]
///
/// use rmatrix_ks::{
///     matrix::{matrix::Matrix, utils::vertical_concat},
///     number::instances::word8::Word8,
/// };
///
/// fn main() {
///     let m = Matrix::<Word8, 2, 2>::of(&[1, 2, 3, 4].map(|e| Word8::of(e))).unwrap();
///     let n = Matrix::<Word8, 2, 2>::of(&[5, 6, 7, 8].map(|e| Word8::of(e))).unwrap();
///     let cat =
///         Matrix::<Word8, 4, 2>::of(&[1, 2, 3, 4, 5, 6, 7, 8].map(|e| Word8::of(e))).unwrap();
///     assert_eq!(vertical_concat(&m, &n), cat);
/// }
/// ```
pub fn vertical_concat<N, const R1: usize, const R2: usize, const C: usize>(
    m1: &Matrix<N, R1, C>,
    m2: &Matrix<N, R2, C>,
) -> Matrix<N, { R1 + R2 }, C>
where
    N: Clone,
{
    let mut inner = Vec::with_capacity((R1 + R2) * C);
    for row_index in 1..=(R1 + R2) {
        for column_index in 1..=C {
            inner.push(if row_index <= R1 {
                m1[(row_index, column_index)].clone()
            } else {
                m2[(row_index - R1, column_index)].clone()
            });
        }
    }
    Matrix { inner }
}

/// Use the Gram-Schmidt process
/// to find the orthogonal basis corresponding to the given set of vectors.
///
/// # Examples
///
/// ```rust
/// use rmatrix_ks::{
///     matrix::{
///         matrix::Matrix,
///         utils::{apply, gram_schmidt_process},
///         vector::dot_product,
///     },
///     number::{
///         instances::float::Float,
///         traits::{one::One, zero::Zero},
///     },
/// };
///
/// fn main() {
///     let basis =
///         Matrix::<Float, 3, 2>::of(&[1.0, 0.0, 1.0, 1.0, 1.0, 1.0].map(Float::of)).unwrap();
///     let ob = gram_schmidt_process(&basis);
///     let c1 = apply(&ob.get_column(1).unwrap(), |e: &Float| e.clone());
///     let c2 = apply(&ob.get_column(2).unwrap(), |e: &Float| e.clone());
///     // Each column vector is normalized.
///     assert_eq!(
///         (dot_product(&c1, &c1), dot_product(&c2, &c2)),
///         (Float::one(), Float::one())
///     );
///     // The column vectors are mutually orthogonal.
///     assert_eq!(dot_product(&c1, &c2), Float::zero());
///     // Corresponding orthogonal basis.
///     let ob_expect = Matrix::<Float, 3, 2>::of(
///         &[
///             1.0 / 3.0f32.sqrt(),
///             -2.0 / 6.0f32.sqrt(),
///             1.0 / 3.0f32.sqrt(),
///             1.0 / 6.0f32.sqrt(),
///             1.0 / 3.0f32.sqrt(),
///             1.0 / 6.0f32.sqrt(),
///         ]
///         .map(Float::of),
///     )
///     .unwrap();
///     assert_eq!(ob, ob_expect);
/// }
/// ```
pub fn gram_schmidt_process<N, const R: usize, const C: usize>(
    basis: &Matrix<N, R, C>,
) -> Matrix<N, R, C>
where
    N: RealFloat,
{
    let mut orthonormal_basis = Matrix::default();
    for k in 1..=C {
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
        for row in 1..=R {
            // OB = [u1 | u2 | ... | uc]
            orthonormal_basis[(row, k)] = uk[(row, 1)].clone();
        }
    }
    orthonormal_basis
}
