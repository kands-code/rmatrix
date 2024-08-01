use rmatrix_ks::error::MatrixError;
use rmatrix_ks::matrix::Matrix;

fn main() -> Result<(), MatrixError> {
    let mat = Matrix::<f32, 3, 3>::create(vec![
        1.0f32, 2.0f32, 4.0f32, 3.0f32, 6.0f32, 8.0f32, 5.0f32, 7.0f32, 9.0f32,
    ])?;
    let eliminates = mat.row_eliminate()?;
    println!("{}", mat);
    println!("{}", eliminates.0);
    println!("{}", eliminates.1);
    println!("{}", eliminates.2);
    println!("{}", mat.row_reduce()?.0);
    println!("{}", mat.row_reduce()?.1);

    Ok(())
}
