use std::f64::consts::PI;

use rand::{thread_rng, Rng};
use rmatrix::matrix::Matrix;

use crate::common::f64_eq;

mod common;

#[test]
fn tr_ab_equals_tr_ba() {
    let times: usize = thread_rng().gen_range(10000..20000);
    let n: usize = thread_rng().gen_range(3..8);
    for _ in 0..times {
        let a = Matrix::rand(n, n, -PI, PI).unwrap();
        let b = Matrix::rand(n, n, -PI, PI).unwrap();
        assert!(f64_eq(
            a.mul(&b).unwrap().trace().unwrap(),
            b.mul(&a).unwrap().trace().unwrap(),
            16
        ))
    }
}
