// Neural network loss functions - MSE and CrossEntropy loss implementations
// Provides Loss trait and specific loss function implementations for training models
// Connected to: src/nn/module.rs, src/nn/functional.rs, src/tensor.rs
// Used by: src/nn/mod.rs, training scripts, tests/test_nn.rs

use crate::nn::module::{Module, ModuleBase, Parameter};
use crate::Tensor;
use std::fmt::{self, Display};

pub trait Loss: Module + Display {
    fn loss(&self, predictions: &Tensor, target: &Tensor) -> Tensor;
}

pub struct MSELoss {
    base: ModuleBase,
}

impl MSELoss {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Display for MSELoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MSELoss()")
    }
}

impl Module for MSELoss {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 2 {
            return Err("MSELoss expects exactly 2 inputs".to_string());
        }
        Ok(self.loss(inputs[0], inputs[1]))
    }
    
    fn parameters(&self) -> Vec<&dyn Parameter> {
        self.base.parameters()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut dyn Parameter> {
        self.base.parameters_mut()
    }
    
    fn modules(&self) -> Vec<&dyn Module> {
        self.base.modules()
    }
    
    fn modules_mut(&mut self) -> Vec<&mut dyn Module> {
        self.base.modules_mut()
    }
    
    fn is_training(&self) -> bool {
        self.base.is_training()
    }
    
    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }
}

impl Loss for MSELoss {
    fn loss(&self, predictions: &Tensor, target: &Tensor) -> Tensor {
        assert_eq!(
            predictions.shape(),
            target.shape(),
            "Labels and predictions shape does not match: {:?} and {:?}",
            predictions.shape(),
            target.shape()
        );
        
        let diff = predictions.sub(target);
        let squared = diff.elementwise_mul(&diff);
        let sum = squared.sum(-1, false);
        sum.scalar_mul(1.0 / predictions.numel as f32)
    }
}

pub struct CrossEntropyLoss {
    base: ModuleBase,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Display for CrossEntropyLoss {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CrossEntropyLoss()")
    }
}

impl Module for CrossEntropyLoss {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 2 {
            return Err("CrossEntropyLoss expects exactly 2 inputs".to_string());
        }
        Ok(self.loss(inputs[0], inputs[1]))
    }
    
    fn parameters(&self) -> Vec<&dyn Parameter> {
        self.base.parameters()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut dyn Parameter> {
        self.base.parameters_mut()
    }
    
    fn modules(&self) -> Vec<&dyn Module> {
        self.base.modules()
    }
    
    fn modules_mut(&mut self) -> Vec<&mut dyn Module> {
        self.base.modules_mut()
    }
    
    fn is_training(&self) -> bool {
        self.base.is_training()
    }
    
    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }
}

impl Loss for CrossEntropyLoss {
    fn loss(&self, input: &Tensor, target: &Tensor) -> Tensor {
        // Use the functional implementation
        crate::nn::functional::cross_entropy_loss(input, target)
    }
}
