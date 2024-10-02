//! # Number Trait :: One
//!
//! Implementing this trait implements the multiplication unit.

/// multiplication unit :: One
pub trait One {
    /// the multiplication unit
    fn one() -> Self;

    /// checks if it is the multiplication unit
    fn is_one(&self) -> bool;
}
