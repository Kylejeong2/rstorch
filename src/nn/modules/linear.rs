use crate::tensor::Tensor;
use crate::nn::module::{Module, ModuleBase, Parameter};
use crate::nn::parameter::ParameterTensor;
use std::fmt;

pub struct Linear {
    base: ModuleBase,
    pub input_dim: usize,
    pub output_dim: usize,
    pub weight: ParameterTensor,
    pub bias: Option<ParameterTensor>,
}

impl Linear { 
    pub fn new(input_dim: usize, output_dim: usize, bias: bool) -> Self {
        let weight = ParameterTensor::new(&[output_dim, input_dim]);
        
        let bias_param = if bias {
            Some(ParameterTensor::new(&[output_dim]))
        } else {
            None
        };

        Self {
            base: ModuleBase::new(),
            input_dim,
            output_dim,
            weight,
            bias: bias_param,
        }
    }
}

impl fmt::Display for Linear {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Linear(in_features={}, out_features={}, bias={})", 
               self.input_dim, self.output_dim, self.bias.is_some())
    }
}

impl Module for Linear {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, String> {
        if inputs.len() != 1 {
            return Err("Linear expects exactly one input".to_string());
        }
        
        let x = inputs[0];
        // Linear layer: y = x @ W^T + b
        // x is [batch, input_dim], weight is [output_dim, input_dim]
        // We need to compute x @ weight.T which gives [batch, output_dim]
        
        // For now, we'll reshape if needed
        let x_2d = if x.ndim == 1 {
            // Convert 1D input to [1, input_dim]
            Tensor::from_vec(x.to_vec(), &[1, x.shape()[0]])
        } else {
            x.clone()
        };
        
        // Transpose weight matrix for multiplication
        // weight is [output_dim, input_dim], we need [input_dim, output_dim]
        let weight_t = self.weight.data().transpose();
        let result = x_2d.matmul(&weight_t);
        
        let result = if let Some(ref bias) = self.bias {
            // Broadcast add bias
            result.add(bias.data())
        } else {
            result
        };
        
        // If input was 1D, output should be 1D
        if x.ndim == 1 {
            Ok(Tensor::from_vec(result.to_vec(), &[result.shape()[1]]))
        } else {
            Ok(result)
        }
    }
    
    fn parameters(&self) -> Vec<&dyn Parameter> {
        let mut params: Vec<&dyn Parameter> = vec![&self.weight];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params.extend(self.base.parameters());
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut dyn Parameter> {
        let mut params: Vec<&mut dyn Parameter> = vec![&mut self.weight];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params.extend(self.base.parameters_mut());
        params
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
    
    fn inner_repr(&self) -> String {
        format!(
            "in_features={}, out_features={}, bias={}",
            self.input_dim,
            self.output_dim,
            self.bias.is_some()
        )
    }
}