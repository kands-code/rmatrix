//! # traits::one
//!
//! Types that implement this trait have the concept of ONE.

/// Concept of ONE.
pub trait One {
    /// Return the ONE number.
    fn one() -> Self;

    /// Validate whether a number is ONE.
    fn is_one(&self) -> bool;
}
