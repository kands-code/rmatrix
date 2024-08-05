use rmatrix_ks::error::Result;
use rmatrix_ks::matrix::Matrix;
use rmatrix_ks::utils::eigen_values;

fn main() -> Result<()> {
    let mat = Matrix::<f32, 2, 2>::create(vec![-3.0, 15.0, 3.0, 9.0])?;

    let eigens = eigen_values(mat, 1024)?;
    eigens.iter().for_each(|e| println!("{}", e));
    Ok(())
}
