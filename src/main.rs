use rmatrix_ks::{
    matrix::matrix::Matrix,
    number::instances::{complex::Complex, float::Float},
};

fn main() {
    let mat: Matrix<Complex<Float>, 2, 2> = Matrix::rand(
        Complex::of(Float::of(-2.0), Float::of(-2.0)),
        Complex::of(Float::of(2.0), Float::of(2.0)),
    );
    println!("{}", mat);
    let trans = mat.transpose();
    println!("{}", trans);
}
