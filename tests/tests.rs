mod common;

use std::f64::consts::PI;

use rand::{thread_rng, Rng};
use rmatrix::{f64_is_zero, matrix::Matrix};

#[test]
fn tr_ab_equals_tr_ba() {
    let times: usize = thread_rng().gen_range(10000..20000);
    let n: usize = thread_rng().gen_range(3..8);
    for _ in 0..times {
        let a = Matrix::<f64>::rand(n, n, -PI, PI).unwrap();
        let b = Matrix::<f64>::rand(n, n, -PI, PI).unwrap();
        assert!(f64_is_zero(
            a.times(&b).unwrap().tr().unwrap() - b.times(&a).unwrap().tr().unwrap()
        ))
    }
}
