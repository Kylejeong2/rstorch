// Main library module for RSTorch - a PyTorch-like deep learning framework in Rust
// This file serves as the root module and re-exports core functionality from submodules
// Connected to: All other modules in the crate (tensor.rs, nn/, optim/, utils/, autograd/, distributed/)
// The library provides tensor operations, neural network layers, optimizers, and distributed training capabilities

pub mod tensor;
pub mod nn;
pub mod optim;
pub mod utils;

pub use tensor::{Tensor, CTensor};
pub mod autograd;
pub mod distributed; 