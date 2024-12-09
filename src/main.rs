use rmatrix_ks::number::{
    instances::{complex::Complex, double::Double},
    traits::floating::Floating,
};

fn main() {
    let cplx = Complex::<Double>::of_str("-2 :+ -4").unwrap();
    println!("{}", cplx.clone().arc_tangent());
}
