use rmatrix_ks::matrix::Matrix;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let m = Matrix::<f64>::from_stdin()?;
    // let a = Matrix::from_vec(
    //     3,
    //     4,
    //     vec![1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 6.0, 1.0, 2.0, 5.0, 2.0],
    // )?;
    // let b = Matrix::from_vec(3, 1, vec![5.0, 10.0, 7.0])?;
    // Matrix::solve_linear_equations(&a, &b)?;
    let a = Matrix::from_vec(3, 3, vec![1.0, 7.0, 3.0, 4.0, 5.0, 9.0, 7.0, 8.0, 9.0])?;
    println!("{}", a.det()?);
    Ok(())
}
