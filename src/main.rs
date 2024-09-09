use rmatrix_ks::{cmplx, error::IResult, num::complex::Complex, num::number::IFractional};
fn main() -> IResult<()> {
    let x = cmplx!(1.0f32, 0.0f32);
    println!("{}", x.nsqrt()?);

    Ok(())
}
