use rmatrix_ks::{error::Result, matrix::Matrix};

fn main() -> Result<()> {
    let mat = Matrix::<f32, 2, 2>::create(vec![1.0, 1.0, 2.0, 4.0])?;
    let elim = mat.row_eliminate()?;

    println!("{}", elim.0);
    println!("{}", elim.1);

    Ok(())
}
