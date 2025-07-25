// Automatic differentiation module - provides gradient computation for backpropagation
// This module implements automatic differentiation for computing gradients during training
// Connected to: src/tensor.rs (GradFn trait implementation), src/nn/ (neural network backward passes)
// Used by: Training loops, optimizer updates, and any code requiring gradient computation

pub mod functions;