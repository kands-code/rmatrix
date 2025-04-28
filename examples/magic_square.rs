use rayon::iter::{ParallelBridge, ParallelIterator};
use rmatrix_ks::{
    matrix::{Matrix, math::is_square_matrix, serde::to_file},
    number::{
        instances::{word::Word, word8::Word8},
        traits::integral::Integral,
        utils::{from_integral, permutation},
    },
};

pub fn is_magic_square<N, const R: usize, const C: usize>(m: &Matrix<N, R, C>) -> bool
where
    N: Integral,
{
    if is_square_matrix(m) {
        let magic_number = R * (R * R + 1) / 2;
        (1..=R).par_bridge().all(|row_index| {
            let sum = (1..=C)
                .par_bridge()
                .map(|column_index| m[(row_index, column_index)].clone())
                .reduce(|| N::zero(), |a, b| a + b);
            from_integral::<Word, N>(sum) == Word::of(magic_number as u32)
        }) && (1..=C).par_bridge().all(|column_index| {
            let sum = (1..=R)
                .par_bridge()
                .map(|row_index| m[(row_index, column_index)].clone())
                .reduce(|| N::zero(), |a, b| a + b);
            from_integral::<Word, N>(sum) == Word::of(magic_number as u32)
        }) && (0..2).par_bridge().all(|p| {
            let sum = (1..=R)
                .par_bridge()
                .map(|e| {
                    if p == 0 {
                        m[(e, e)].clone()
                    } else {
                        m[(e, R - e + 1)].clone()
                    }
                })
                .reduce(|| N::zero(), |a, b| a + b);
            from_integral::<Word, N>(sum) == Word::of(magic_number as u32)
        })
    } else {
        eprintln!(concat!(
            "[Error::magic_square]: ",
            "Only square matrices can possibly be magic squares."
        ));
        false
    }
}

fn main() {
    let one_to_nine = [1, 2, 3, 4, 5, 6, 7, 8, 9].map(Word8::of).to_vec();
    let magic_square = permutation(&one_to_nine)
        .iter()
        .map(|p| Matrix::<Word8, 3, 3>::of(p).unwrap())
        .filter(is_magic_square)
        .take(1)
        .collect::<Vec<Matrix<Word8, 3, 3>>>()[0]
        .clone();
    to_file(&magic_square, "data/magic_square.txt".to_string());
}
