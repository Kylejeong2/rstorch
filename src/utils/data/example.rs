// Training example data structure - Basic data container for individual training samples
// Provides minimal Example struct that holds feature data for training/inference
// Connected to: src/utils/data/batch.rs, src/utils/data/dataset.rs, src/utils/data/field.rs
// Used by: src/utils/data/batch.rs, src/utils/data/dataset.rs, src/utils/data/dataloader.rs

/// A minimal `Example` placeholder. In a real implementation this could be a more
/// complex struct representing a single training example (e.g. features + label).
#[derive(Clone, Debug)]
pub struct Example {
    pub data: Vec<f32>,
} 