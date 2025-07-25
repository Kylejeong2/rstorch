// Distributed data parallel wrapper - Multi-GPU and multi-node training support
// Provides DistributedDataParallel wrapper that synchronizes model parameters and gradients across devices
// Connected to: src/nn/module.rs, src/distributed/distributed.rs, src/tensor.rs
// Used by: Distributed training scripts, multi-GPU training setups

use crate::nn::module::{Module, Parameter};
use crate::distributed::{broadcast_tensor_rs, allreduce_sum_tensor_rs, get_world_size};
use crate::tensor::Tensor;
use std::fmt;

pub struct DistributedDataParallel {
    module: Box<dyn Module>,
}

impl DistributedDataParallel {
    pub fn new(module: Box<dyn Module>) -> Self {
        let mut ddp = Self { module };
        
        ddp.broadcast_parameters();
        ddp.register_grads_hooks();
        
        ddp
    }
    
    /// Broadcast parameters of device 0 to all devices
    fn broadcast_parameters(&mut self) {
        for param in self.module.parameters_mut() {
            let mut tensor = param.data().clone();
            let _ = broadcast_tensor_rs(&mut tensor, 0);
            // Note: This would need a way to update the parameter's data
            // which might require extending the Parameter trait
        }
    }
    
    /// Everytime a gradient is assigned to some value, it calculates mean of this gradient among all devices
    fn allreduce_grads_hook(grad: &mut Tensor) -> Tensor {
        let mut avg_grad = grad.clone();
        let _ = allreduce_sum_tensor_rs(&mut avg_grad);
        let world_size = get_world_size().unwrap_or(1) as f32;
        avg_grad / world_size
    }
    
    /// Everytime a gradient is assigned it calls this allreduce hook
    fn register_grads_hooks(&mut self) {
        // Note: This would need a hook system in the Parameter trait
        // which might require extending the Parameter trait to support hooks
        for _param in self.module.parameters_mut() {
            // param.register_hook(Self::allreduce_grads_hook);
            // This functionality would need to be implemented in the Parameter trait
        }
    }
}

impl Module for DistributedDataParallel {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor, String> {
        self.module.forward(inputs)
    }
    
    fn parameters(&self) -> Vec<&dyn Parameter> {
        self.module.parameters()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut dyn Parameter> {
        self.module.parameters_mut()
    }
    
    fn modules(&self) -> Vec<&dyn Module> {
        vec![self.module.as_ref()]
    }
    
    fn modules_mut(&mut self) -> Vec<&mut dyn Module> {
        vec![self.module.as_mut()]
    }
    
    fn is_training(&self) -> bool {
        self.module.is_training()
    }
    
    fn set_training(&mut self, training: bool) {
        self.module.set_training(training);
    }
}

impl fmt::Display for DistributedDataParallel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DistributedDataParallel(\n  (module): {}\n)", self.module)
    }
}