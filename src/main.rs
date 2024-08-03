use rmatrix_ks::error::MatrixError;
use rmatrix_ks::matrix::Matrix;

fn main() -> Result<(), MatrixError> {
    let mat = Matrix::<f32, 3, 4>::create(vec![
        1.0f32, 2.0f32, 3.0f32, -1.0f32, 4.0f32, 5.0f32, 6.0f32, 2.0f32, 7.0f32, 8.0f32, 9.0f32,
        3.0f32,
    ])?;
    println!("{}", mat);
    println!("{}", mat.submatrix(1, 4)?);
    println!("{}", mat.get_diag()?);

    Ok(())
}
