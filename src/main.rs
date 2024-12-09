use rmatrix_ks::matrix::matrix::Matrix;

fn main() {
    let mat: Matrix<u8, 2, 2> = Matrix::of(&[1, 2, 3, 4]).unwrap();
    println!("{}", mat);
}
