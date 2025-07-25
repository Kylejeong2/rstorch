// Neural network module - provides layers, activation functions, loss functions, and training utilities
// This module contains all neural network components including layers, optimizers, and utilities
// Connected to: src/tensor.rs (tensor operations), src/autograd/ (backpropagation), src/optim/ (optimizers)
// Used by: Training scripts, model definitions, and the distributed training system

pub mod module;
pub mod parameter;
pub mod modules;
pub mod activation;
pub mod loss;
pub mod functional;
pub mod parallel;

// Re-exports for convenience
pub use module::{Module, Parameter, ModuleBase};
pub use parameter::ParameterTensor;
pub use activation::{Sigmoid, Softmax, ReLU, Activation};
pub use loss::{MSELoss, CrossEntropyLoss, Loss};
pub use modules::linear::Linear;
pub use parallel::DistributedDataParallel;