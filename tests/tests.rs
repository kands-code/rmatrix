mod common;

use std::f64::consts::PI;

use rand::{thread_rng, Rng};
use rmatrix::{matrix::Matrix, number::Number};

#[test]
fn tr_ab_equals_tr_ba() {
    let times: usize = thread_rng().gen_range(10000..20000);
    let n: usize = thread_rng().gen_range(3..8);
    for _ in 0..times {
        let a = Matrix::<f64>::rand(n, n, -PI, PI).unwrap();
        let b = Matrix::<f64>::rand(n, n, -PI, PI).unwrap();
        let r = a.times(&b).unwrap().tr().unwrap() - b.times(&a).unwrap().tr().unwrap();
        assert!(r.is_zero(), "{}?", r)
    }
}
