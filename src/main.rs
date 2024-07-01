use rmatrix_ks::error::MatrixError;
use rmatrix_ks::matrix::Matrix;

fn main() -> Result<(), MatrixError> {
    let mat: Matrix<i8, 2, 3> = Matrix::create(vec![1, 2, 3, 4, 5, 6])?;
    println!("{}", mat);
    let diag = mat.get_diag()?;
    println!("{}", diag);
    Ok(())
}
