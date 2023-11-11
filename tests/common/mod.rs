pub fn f64_eq(f1: f64, f2: f64, s: u8) -> bool {
    ((f1 * 10.0f64.powi(s as i32)).trunc() as i32) - ((f2 * 10.0f64.powi(s as i32)).trunc() as i32)
        == 0
}
