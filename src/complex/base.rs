use crate::{complex::Complex, f64_is_zero};

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Complex(re, im)
    }

    pub fn re(&self) -> f64 {
        self.0
    }

    pub fn im(&self) -> f64 {
        self.1
    }

    pub fn norm(&self) -> f64 {
        (self.re().powi(2) + self.im().powi(2)).sqrt()
    }

    pub fn is_zero(&self) -> bool {
        f64_is_zero(self.norm())
    }
}
