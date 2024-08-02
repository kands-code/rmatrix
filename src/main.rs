use rmatrix_ks::error::MatrixError;
use rmatrix_ks::matrix::Matrix;
use rmatrix_ks::rat;
use rmatrix_ks::rational::Rational;

fn main() -> Result<(), MatrixError> {
    let mat = Matrix::<Rational<i32>, 2, 2>::create(vec![
        rat!(1, 1),
        rat!(2, 1),
        rat!(3, 1),
        rat!(4, 1),
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
