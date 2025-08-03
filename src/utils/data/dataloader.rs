// DataLoader iterator - Converts datasets into mini-batch iterators for training
// Provides DataLoader that creates shuffled mini-batches from datasets with configurable batch size
// Connected to: src/utils/data/dataset.rs, src/utils/data/batch.rs
// Used by: Training loops, model training scripts, examples

use rand::seq::SliceRandom;

use super::batch::Batch;
use super::dataset::Dataset;

/// Trait for sampling indices from a dataset
pub trait Sampler {
    /// Generate indices for the given dataset length
    fn sample(&self, dataset_len: usize) -> Vec<usize>;
}

/// Sequential sampler that returns indices in order
#[derive(Clone, Debug)]
pub struct SequentialSampler;

impl Sampler for SequentialSampler {
    fn sample(&self, dataset_len: usize) -> Vec<usize> {
        (0..dataset_len).collect()
    }
}

/// Random sampler that shuffles indices
#[derive(Clone, Debug)]
pub struct RandomSampler;

impl Sampler for RandomSampler {
    fn sample(&self, dataset_len: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..dataset_len).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        indices
    }
}

/**
Turns a dataset into an iterator of mini-batches ready for training 
*/

#[derive(Clone, Debug)]
pub struct DataLoader<'a> {
    dataset: &'a Dataset,
    batch_size: usize,
    indices: Vec<usize>,
    cursor: usize,
}

impl<'a> DataLoader<'a> { 
    // <'a> is lifetime param, doesn't cost anything since it is enforced by the compiler
    /// Create a new DataLoader with a sampler.
    pub fn new_with_sampler(dataset: &'a Dataset, batch_size: usize, sampler: Box<dyn Sampler>) -> Self {
        let indices = sampler.sample(dataset.len());
        Self {
            dataset,
            batch_size,
            indices,
            cursor: 0,
        }
    }

    /// Create a new DataLoader with optional random shuffling.
    pub fn new(dataset: &'a Dataset, batch_size: usize, shuffle: bool) -> Self {
        let sampler: Box<dyn Sampler> = if shuffle {
            Box::new(RandomSampler)
        } else {
            Box::new(SequentialSampler)
        };
        Self::new_with_sampler(dataset, batch_size, sampler)
    }

    /// Number of batches this loader will produce.
    pub fn len(&self) -> usize {
        (self.dataset.len() + self.batch_size - 1) / self.batch_size
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.indices.len() {
            return None;
        }
        let end = (self.cursor + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.cursor..end];
        self.cursor = end;

        let data: Vec<_> = batch_indices
            .iter()
            .filter_map(|&idx| self.dataset.get(idx).cloned())
            .collect();
        Some(Batch::new(data))
    }
}