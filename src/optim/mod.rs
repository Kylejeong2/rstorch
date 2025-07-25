// Optimization module - provides optimizers for training neural networks
// This module contains optimizer implementations for updating model parameters during training
// Connected to: src/nn/module.rs (accessing parameters), src/tensor.rs (parameter updates), training loops
// Contains: SGD, Adam, AdamW, and other optimization algorithms

pub mod optimizer;
pub mod optimizers;

pub use optimizer::{Optimizer, OptimizerParams};
pub use optimizers::SGD;