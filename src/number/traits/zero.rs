//! # Number Trait :: Zero
//!
//! Implementing this trait implements the additive unit.

/// additive unit :: Zero
pub trait Zero {
    /// the additive unit
    fn zero() -> Self;

    /// checks if it is the additive unit
    fn is_zero(&self) -> bool;
}
