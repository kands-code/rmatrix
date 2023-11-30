use rmatrix_ks::matrix::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let m = Matrix::<f64>::from_stdin()?;
    let m = Matrix::from_vec(3, 5, vec![1, 1, 2, 1, 5, 1, 1, 2, 6, 10, 1, 2, 5, 2, 7])?;
    println!("{}", m.row_reduce()?);
    Ok(())
}
