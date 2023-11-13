use rmatrix::{complex::Complex, matrix::Matrix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m = Matrix::<Complex>::from_stdin()?;
    println!("{}", m);

    Ok(())
}
