// Optimizer trait and parameter management for neural network training
// This file defines the Optimizer trait and OptimizerParams for managing trainable parameters
// Connected to: src/tensor.rs (parameter tensors), src/nn/module.rs (model parameters), src/optim/optimizers/ (concrete optimizer implementations)
// Used by: Training loops, concrete optimizers (SGD, Adam, etc.)

use crate::tensor::Tensor;
use std::collections::HashMap;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

pub struct OptimizerParams {
    pub parameters: Vec<(String, String, Tensor)>, // (module, name, parameter)
}

impl OptimizerParams {
    pub fn new<P>(parameters: P) -> Result<Self, String> 
    where
        P: IntoIterator<Item = (String, String, Tensor)>
    {
        let params: Vec<(String, String, Tensor)> = parameters.into_iter().collect();
        Ok(OptimizerParams { parameters: params })
    }

    pub fn from_dict(parameters: HashMap<String, Tensor>) -> Self {
        let params = parameters
            .into_iter()
            .map(|(name, tensor)| (String::new(), name, tensor))
            .collect();
        OptimizerParams { parameters: params }
    }

    pub fn zero_grad(&mut self) {
        for (_, _, parameter) in &mut self.parameters {
            parameter.zero_grad();
        }
    }
}