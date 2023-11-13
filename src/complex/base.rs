//! basic implementation of Complex

use crate::complex::Complex;

impl Complex {
    /// generate a complex number
    pub fn new(re: f64, im: f64) -> Self {
        Complex(re, im)
    }

    /// get the real part of a complex number
    pub fn re(&self) -> f64 {
        self.0
    }

    /// get the imagine part of a complex number
    pub fn im(&self) -> f64 {
        self.1
    }

    /// get the norm of a complex number
    pub fn norm(&self) -> f64 {
        (self.re().powi(2) + self.im().powi(2)).sqrt()
    }
}
