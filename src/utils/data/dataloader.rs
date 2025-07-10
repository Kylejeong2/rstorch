use rand::seq::SliceRandom;

use super::batch::Batch;
use super::dataset::Dataset;

/**
Turns a dataset into an iterator of mini-batches ready for training 
*/

/// Optional sampler: if None, sequential indices are used; otherwise a precomputed
/// vector of indices (e.g. shuffled) is consumed.

// TODO: add a sampler trait

#[derive(Clone, Debug)]
pub struct DataLoader<'a> {
    dataset: &'a Dataset,
    batch_size: usize,
    indices: Vec<usize>,
    cursor: usize,
}

impl<'a> DataLoader<'a> { 
    // <'a> is lifetime param, doesn't cost anything since it is enforced by the compiler
    /// Create a new DataLoader with optional random shuffling.
    pub fn new(dataset: &'a Dataset, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.len()).collect();
        if shuffle {
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
        Self {
            dataset,
            batch_size,
            indices,
            cursor: 0,
        }
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
        let batch_idxs = &self.indices[self.cursor..end];
        self.cursor = end;

        let data: Vec<_> = batch_idxs
            .iter()
            .filter_map(|&idx| self.dataset.get(idx).cloned())
            .collect();
        Some(Batch::new(data))
    }
}