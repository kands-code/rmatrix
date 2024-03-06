use rmatrix_ks::{complex::Complex, file::IFile, Matrix};
use std::{fs::File, io::read_to_string};

fn main() {
    let m = Matrix::<_, 3, 3>::diag(&vec![Complex(2, 3), Complex(4, 2), Complex(5, 1)]);
    println!("{}", m);
    Matrix::write_to("data/test.json", &vec![m]);
    let file = File::open("data/test.json").unwrap();
    let vm: Vec<Matrix<Complex<i32>, 3, 3>> = Matrix::read_from(&read_to_string(file).unwrap());
    println!("{}", vm[0]);
}
