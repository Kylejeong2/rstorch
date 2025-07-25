// Batch data structure - Container for grouped training examples
// Provides Batch struct for managing collections of examples during mini-batch training
// Connected to: src/utils/data/example.rs, src/utils/data/dataloader.rs
// Used by: src/utils/data/dataloader.rs, training loops, model forward passes

use super::example::Example;

/// A batch of examples with helper methods.
#[derive(Clone, Debug)]
pub struct Batch {
    pub data: Vec<Example>,
}

impl Batch {
    pub fn new(data: Vec<Example>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, idx: usize) -> Option<&Example> {
        self.data.get(idx)
    }
}