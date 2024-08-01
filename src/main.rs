use rmatrix_ks::error::MatrixError;
use rmatrix_ks::matrix::Matrix;

fn main() -> Result<(), MatrixError> {
    let mat = Matrix::<f64, 3, 4>::create(vec![
        1.0, 2.0, 3.0, -1.0, 4.0, 5.0, 6.0, 2.0, 7.0, 8.0, 9.0, 3.0,
    ])?;

    println!("{}", mat);
    println!("{}", mat.row_eliminate()?.0);

    Ok(())
}
