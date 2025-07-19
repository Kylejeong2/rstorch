use crate::tensor::Tensor;
use crate::nn::module::{Module, ModuleBase};
use std::fmt;

pub trait Activation: Module {
}

pub struct Sigmoid { //maps values to 0-1
    base: ModuleBase,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Sigmoid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sigmoid()")
    }
}

impl Module for Sigmoid {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("Sigmoid expects exactly one input".to_string());
        }
        Ok(inputs[0].sigmoid())
    }
    
    fn parameters(&self) -> Vec<&dyn crate::nn::module::Parameter> {
        self.base.parameters()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut dyn crate::nn::module::Parameter> {
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

impl Activation for Sigmoid {}

pub struct Softmax {
    base: ModuleBase,
    dim: i32,
}

impl Softmax {
    pub fn new(dim: i32) -> Self {
        Self {
            base: ModuleBase::new(),
            dim,
        }
    }
}

impl fmt::Display for Softmax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Softmax(dim={})", self.dim)
    }
}

impl Module for Softmax {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("Softmax expects exactly one input".to_string());
        }
        Ok(inputs[0].softmax(self.dim))
    }
    
    fn parameters(&self) -> Vec<&dyn crate::nn::module::Parameter> {
        self.base.parameters()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut dyn crate::nn::module::Parameter> {
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

impl Activation for Softmax {}

pub struct ReLU {
    base: ModuleBase,
}

impl ReLU {
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
        }
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ReLU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ReLU()")
    }
}

impl Module for ReLU {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("ReLU expects exactly one input".to_string());
        }
        Ok(inputs[0].relu())
    }
    
    fn parameters(&self) -> Vec<&dyn crate::nn::module::Parameter> {
        self.base.parameters()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut dyn crate::nn::module::Parameter> {
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

impl Activation for ReLU {}
