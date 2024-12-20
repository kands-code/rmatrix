//! # traits::zero
//!
//! Types that implement this trait have the concept of ZERO.

/// Concept of ZERO.
pub trait Zero {
    /// Return the ZERO number.
    fn zero() -> Self;

    /// Validate whether a number is ZERO.
    fn is_zero(&self) -> bool;
}
