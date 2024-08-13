use rmatrix_ks::error::IResult;
use rmatrix_ks::matrix::Matrix;
use rmatrix_ks::utils::decompose::qr_decomposition_reduced;

fn main() -> IResult<()> {
    let mat = Matrix::<f32>::create(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    println!("{}", mat);

    let qr = qr_decomposition_reduced(mat)?;
    println!("{}", qr.0);
    println!("{}", qr.1);

    Ok(())
}
