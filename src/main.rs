use rmatrix_ks::error::MatrixError;
use rmatrix_ks::matrix::Matrix;
use rmatrix_ks::rat;
use rmatrix_ks::rational::Rational;
use rmatrix_ks::utils::lu_decomposition;

fn main() -> Result<(), MatrixError> {
    let mat = Matrix::<Rational<i128>, 3, 3>::create(vec![
        rat!(25, 1),
        rat!(5, 1),
        rat!(1, 1),
        rat!(64, 1),
        rat!(8, 1),
        rat!(1, 1),
        rat!(144, 1),
        rat!(12, 1),
        rat!(1, 1),
    ])?;
    println!("{}", mat);
    let reduced = lu_decomposition(mat)?;
    println!("{}", reduced.0);
    println!("{}", reduced.1);

    Ok(())
}
