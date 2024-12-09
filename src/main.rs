use rmatrix_ks::number::{
    instances::{complex::Complex, double::Double},
    traits::floating::Floating,
};

fn main() {
    let cplx = Complex::of(Double::of(1.0f64), Double::of(2.0f64));
    println!("{}", cplx.clone().arc_hyperbolic_sine());
}
