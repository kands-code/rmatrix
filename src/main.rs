use rmatrix_ks::{
    error::Result,
    matrix::Matrix,
    utils::decompose::{qr_decomposition, qr_decomposition_gs},
};

fn main() -> Result<()> {
    let mat = Matrix::<f32, 4, 3>::create(vec![
        1.0, -1.0, 4.0, 1.0, 4.0, -2.0, 1.0, 4.0, 2.0, 1.0, -1.0, 0.0,
    ])?;
    println!("{}", mat);
    let qr = qr_decomposition(mat)?;
    println!("{}", qr.0);
    println!("{}", qr.1);

    let mat = Matrix::<f32, 3, 3>::create(vec![1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])?;
    let qr_gs = qr_decomposition_gs(mat.to_owned())?;
    let qr_hh = qr_decomposition(mat)?;
    println!("{}", qr_gs.0);
    println!("{}", qr_gs.1);
    println!("{}", qr_hh.0);
    println!("{}", qr_hh.1);

    Ok(())
}
