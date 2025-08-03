// Stochastic Gradient Descent optimizer - SGD with momentum support
// Implements the SGD optimization algorithm for neural network parameter updates
// Connected to: src/optim/optimizer.rs, src/tensor.rs
// Used by: src/optim/optimizers/mod.rs, training scripts, examples

use crate::optim::optimizer::{Optimizer, OptimizerParams};
use crate::tensor::Tensor;
use std::collections::HashMap;

pub struct SGD {
    pub params: OptimizerParams,
    pub lr: f32,
    pub momentum: f32,
    pub velocity_cache: Vec<Tensor>,
}

impl SGD {
    pub fn new(parameters: OptimizerParams, lr: f32, momentum: f32) -> Self {
        let velocity_cache = parameters.parameters
            .iter()
            .map(|(_, _, param)| param.zeros_like().expect("Failed to create zeros_like tensor"))
            .collect();
        
        Self {
            params: parameters,
            lr,
            momentum,
            velocity_cache,
        }
    }
    
    pub fn from_dict(parameters: HashMap<String, Tensor>, lr: f32, momentum: f32) -> Self {
        let params = OptimizerParams::from_dict(parameters);
        Self::new(params, lr, momentum)
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (i, (_, _, parameter)) in self.params.parameters.iter_mut().enumerate() {
            if let Some(grad) = &parameter.grad {
                let velocity = &mut self.velocity_cache[i];
                
                // velocity = momentum * velocity - lr * grad
                *velocity = velocity.scalar_mul(self.momentum).add(&grad.scalar_mul(-self.lr));
                
                // parameter = parameter + velocity
                *parameter = parameter.add(velocity);
            }
        }
    }
    
    fn zero_grad(&mut self) {
        self.params.zero_grad();
    }
}