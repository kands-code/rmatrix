//! # State
//!
//! use state to distinguish the shape of the matrix,
//! mainly used to distinguish the direction of the vector

/// state for matrices
#[derive(Debug, Clone)]
pub struct SMatrix;

/// state for row vectors
#[derive(Debug, Clone)]
pub struct SRow;

/// state for column vectors
#[derive(Debug, Clone)]
pub struct SColumn;
