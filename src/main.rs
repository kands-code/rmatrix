use rmatrix_ks::matrix::Matrix;
use rmatrix_ks::utils::plu_decomposition;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mat = Matrix::<f64, 3, 3>::create(vec![1.0, 2.0, 3.0, 4.0, 8.0, 7.0, 5.0, 9.0, 6.0])?;
    println!("{}", mat);
    let plu = plu_decomposition(mat.to_owned())?;
    println!("{}", plu.0);
    println!("{}", plu.1);
    println!("{}", plu.2);
    println!("{}", mat.determinant()?);

    Ok(())
}
