pub mod complex;
pub mod error;
pub mod matrix;
pub mod types;

pub fn f64_is_zero(f64num: f64) -> bool {
    (f64num).abs() < std::f64::EPSILON
}
