use rmatrix::core::Matrix;

fn main() {
    let mut v = Vec::new();
    for _ in 0..10 {
        v.push(Matrix::rand(3, 3, -std::f64::consts::PI, std::f64::consts::PI).unwrap());
    }
    Matrix::to_json(&v, "data/mats.json").unwrap();

    let json_v = Matrix::from_json("data/mats.json").unwrap();
    for m in json_v {
        println!("{}", m);
    }
}
