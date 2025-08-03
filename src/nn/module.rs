// Neural network module trait and base implementation - defines the interface for neural network layers
// This file provides the Module trait that all neural network layers implement, along with parameter management
// Connected to: src/tensor.rs (tensor operations), src/nn/modules/ (concrete layer implementations), src/optim/ (optimizer access to parameters)
// Used by: All neural network layers (Linear, Conv2d, etc.), training loops, and model serialization

use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fmt;

pub trait Parameter {
    fn zero_grad(&mut self);
    fn requires_grad(&self) -> bool;
    fn set_requires_grad(&mut self, requires_grad: bool);
    fn to_device(&mut self, _device: &str);
    fn shape(&self) -> &[usize];
    fn data(&self) -> &Tensor;
    fn data_mut(&mut self) -> &mut Tensor;
    fn set_data(&mut self, data: Vec<f32>) -> Result<(), String>;
}

pub trait Module: fmt::Display {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, String>;
    
    fn call(&self, inputs: &[&Tensor]) -> Result<Tensor, String> {
        self.forward(inputs)
    }
    
    fn train(&mut self) {
        self.set_training(true);
        // Set requires_grad for all parameters to true
        // This would need to be implemented in each concrete module type
    }
    
    fn eval(&mut self) {
        self.set_training(false);
        // Set requires_grad for all parameters to false
        // This would need to be implemented in each concrete module type
    }
    
    fn parameters(&self) -> Vec<&dyn Parameter>;
    fn parameters_mut(&mut self) -> Vec<&mut dyn Parameter>;
    
    fn modules(&self) -> Vec<&dyn Module>;
    fn modules_mut(&mut self) -> Vec<&mut dyn Module>;
    
    fn zero_grad(&mut self) {
        // Default implementation does nothing
        // Concrete implementations should override this
    }
    
    fn to_device(&mut self, _device: &str) {
        // Default implementation does nothing
        // Concrete implementations should override this
    }
    
    fn state_dict(&self) -> HashMap<String, Vec<f32>> {
        let mut state = HashMap::new();
        for (i, param) in self.parameters().iter().enumerate() {
            let key = format!("param{}", i);
            state.insert(key, param.data().to_vec());
        }
        state
    }
    
    fn load_state_dict(&mut self, state_dict: &HashMap<String, Vec<f32>>) -> Result<(), String> {
        for (i, param) in self.parameters_mut().iter_mut().enumerate() {
            let key = format!("param{}", i);
            if let Some(data) = state_dict.get(&key) {
                if param.shape().iter().product::<usize>() != data.len() {
                    return Err(format!("Shape mismatch for parameter {}: expected {}, got {}", 
                                     i, param.shape().iter().product::<usize>(), data.len()));
                }
                param.set_data(data.clone())?;
            }
        }
        Ok(())
    }
    
    fn save(&self, filename: &str) -> Result<(), std::io::Error> {
        let state = self.state_dict();
        let serialized = serde_json::to_string(&state)?;
        std::fs::write(filename, serialized)
    }
    
    fn is_training(&self) -> bool;
    fn set_training(&mut self, training: bool);
    
    fn module_name(&self) -> &'static str {
        std::any::type_name::<Self>().split("::").last().unwrap_or("Module")
    }
    
    fn inner_repr(&self) -> String {
        String::new()
    }
}

pub struct ModuleBase {
    training: bool,
    modules: HashMap<String, Box<dyn Module>>,
    parameters: HashMap<String, Box<dyn Parameter>>,
}

impl ModuleBase {
    pub fn new() -> Self {
        Self {
            training: true,
            modules: HashMap::new(),
            parameters: HashMap::new(),
        }
    }
    
    pub fn add_module(&mut self, name: String, module: Box<dyn Module>) {
        self.modules.insert(name, module);
    }
    
    pub fn add_parameter(&mut self, name: String, parameter: Box<dyn Parameter>) {
        self.parameters.insert(name, parameter);
    }
}

impl Default for ModuleBase {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ModuleBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(", self.module_name())?;
        if self.modules.is_empty() {
            write!(f, "\n   (parameters): {}", <ModuleBase as Module>::inner_repr(self))?;
        } else {
            for (key, module) in &self.modules {
                write!(f, "\n   ({}): {}", key, module)?;
            }
        }
        write!(f, "\n)")
    }
}

impl Module for ModuleBase {
    fn forward(&self, _inputs: &[&Tensor]) -> Result<Tensor, String> {
        Err("forward method not implemented".to_string())
    }
    
    fn parameters(&self) -> Vec<&dyn Parameter> {
        let mut params = Vec::new();
        for param in self.parameters.values() {
            params.push(param.as_ref());
        }
        for module in self.modules.values() {
            params.extend(module.parameters());
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut dyn Parameter> {
        // This is a placeholder implementation that returns an empty vec
        // The actual implementation would require unsafe code or a different design
        // to handle the lifetime issues with mutable references
        Vec::new()
    }
    
    fn modules(&self) -> Vec<&dyn Module> {
        self.modules.values().map(|m| m.as_ref()).collect()
    }
    
    fn modules_mut(&mut self) -> Vec<&mut dyn Module> {
        // This is a placeholder implementation that returns an empty vec
        // The actual implementation would require unsafe code or a different design
        // to handle the lifetime issues with mutable references
        Vec::new()
    }
    
    fn is_training(&self) -> bool {
        self.training
    }
    
    fn set_training(&mut self, training: bool) {
        self.training = training;
        for module in self.modules.values_mut() {
            module.set_training(training);
        }
    }
    
    fn zero_grad(&mut self) {
        // Zero gradients in self parameters
        for param in self.parameters.values_mut() {
            param.zero_grad();
        }
        // Zero gradients in child modules
        for module in self.modules.values_mut() {
            module.zero_grad();
        }
    }
    
    fn to_device(&mut self, device: &str) {
        // Move self parameters to device
        for param in self.parameters.values_mut() {
            param.to_device(device);
        }
        // Move child modules to device
        for module in self.modules.values_mut() {
            module.to_device(device);
        }
    }
}

