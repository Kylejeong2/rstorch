use super::dataset::Dataset;

/// Sampler that splits dataset indices across multiple processes for
/// data-parallel training (similar to PyTorch `DistributedSampler`).
///
/// It ensures every replica (process) sees a unique contiguous subset of
/// the indices but the union still covers the whole (possibly padded)
/// dataset. Padding duplicates the first samples to make the length
/// divisible by `num_replicas` 
#[derive(Debug, Clone)]
pub struct DistributedSampler {
    dataset_len: usize,
    num_replicas: usize,
    rank: usize,
    num_samples: usize,
    total_size: usize,
}

impl DistributedSampler {
    /// Create a new sampler.
    ///
    /// * `dataset` – any type that exposes `len()`
    /// * `num_replicas` – world size (total processes)
    /// * `rank` – id of current process (0-based)
    pub fn new(dataset: &Dataset, num_replicas: usize, rank: usize) -> Self {
        assert!(rank < num_replicas, "rank must be in 0..num_replicas");
        let dataset_len = dataset.len();
        let num_samples = ((dataset_len as f32) / num_replicas as f32).ceil() as usize;
        let total_size = num_samples * num_replicas;
        Self {
            dataset_len,
            num_replicas,
            rank,
            num_samples,
            total_size,
        }
    }

    /// Return the (sub)indices assigned to this replica.
    pub fn indices(&self) -> Vec<usize> {
        // 1. base indices 0..len
        let mut all_indices: Vec<usize> = (0..self.dataset_len).collect();
        // 2. Pad / truncate to total_size so it's divisible.
        if self.total_size > all_indices.len() {
            let extra = self.total_size - all_indices.len();
            let padding: Vec<usize> = all_indices[..extra].to_vec();
            all_indices.extend(padding);
        } else if self.total_size < all_indices.len() {
            all_indices.truncate(self.total_size);
        }
        debug_assert_eq!(all_indices.len(), self.total_size);

        // 3. Sub-sample for this rank.
        all_indices
            .iter()
            .skip(self.rank)
            .step_by(self.num_replicas)
            .copied()
            .collect()
    }

    /// Number of samples for this replica (length of `indices()` vector).
    pub fn len(&self) -> usize {
        self.num_samples
    }
}

impl IntoIterator for DistributedSampler {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.indices().into_iter()
    }
} 