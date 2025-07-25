// Optimizers module - Exports optimization algorithms
// Aggregates all optimizer implementations for model training
// Connected to: src/optim/optimizers/sgd.rs
// Used by: src/optim/mod.rs, training scripts, examples

pub mod sgd;

pub use sgd::SGD;