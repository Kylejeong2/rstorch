use rstorch::utils::data::{Dataset, Example, Field, Batch, DataLoader, DistributedSampler};

fn create_test_dataset() -> Dataset {
    let mut examples = Vec::new();
    for i in 0..10 {
        // Create a simple feature vector with id, value, and label
        let data = vec![i as f32, i as f32 * 2.0, (i % 2) as f32];
        examples.push(Example { data });
    }
    
    let fields = vec![
        Field {
            name: "id".to_string(),
            dtype: "int".to_string(),
            transform: None,
        },
        Field {
            name: "value".to_string(),
            dtype: "float".to_string(),
            transform: None,
        },
        Field {
            name: "label".to_string(),
            dtype: "int".to_string(),
            transform: None,
        },
    ];
    
    Dataset::new(examples, fields)
}

#[test]
fn test_dataset_creation() {
    let dataset = create_test_dataset();
    assert_eq!(dataset.len(), 10);
    assert_eq!(dataset.fields.len(), 3);
}

#[test]
fn test_dataset_indexing() {
    let dataset = create_test_dataset();
    
    let example = dataset.get(0).unwrap();
    assert_eq!(example.data[0], 0.0);  // id
    assert_eq!(example.data[1], 0.0);  // value
    assert_eq!(example.data[2], 0.0);  // label
    
    let example = dataset.get(5).unwrap();
    assert_eq!(example.data[0], 5.0);  // id
    assert_eq!(example.data[1], 10.0); // value
    assert_eq!(example.data[2], 1.0);  // label
}

#[test]
fn test_dataset_slicing() {
    let dataset = create_test_dataset();
    // Create a slice manually since Dataset doesn't have a slice method
    let slice_examples = dataset.examples[2..5].to_vec();
    let slice_dataset = Dataset::new(slice_examples, dataset.fields.clone());
    
    assert_eq!(slice_dataset.len(), 3);
    assert_eq!(slice_dataset.get(0).unwrap().data[0], 2.0);
    assert_eq!(slice_dataset.get(1).unwrap().data[0], 3.0);
    assert_eq!(slice_dataset.get(2).unwrap().data[0], 4.0);
}

#[test]
fn test_dataset_split() {
    let dataset = create_test_dataset();
    // Use Dataset::splits which expects train and valid ratios
    let (train, valid, test) = Dataset::splits(dataset.examples.clone(), dataset.fields.clone(), 0.7, 0.2);
    
    assert_eq!(train.len(), 7);
    assert_eq!(valid.len(), 2);
    assert_eq!(test.len(), 1);
    
    // Check that examples are split correctly
    for i in 0..7 {
        assert_eq!(train.get(i).unwrap().data[0], i as f32);
    }
    for i in 0..2 {
        assert_eq!(valid.get(i).unwrap().data[0], (i + 7) as f32);
    }
}

#[test]
fn test_batch_creation() {
    let examples = vec![
        Example { data: vec![1.0, 2.0, 3.0] },
        Example { data: vec![4.0, 5.0, 6.0] },
    ];
    
    let batch = Batch::new(examples.clone());
    assert_eq!(batch.len(), 2);
    
    // Check batched data
    assert_eq!(batch.get(0).unwrap().data, vec![1.0, 2.0, 3.0]);
    assert_eq!(batch.get(1).unwrap().data, vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_dataloader_basic() {
    let dataset = create_test_dataset();
    let dataloader = DataLoader::new(&dataset, 3, true);
    
    let batches: Vec<_> = dataloader.collect();
    assert_eq!(batches.len(), 4); // 10 examples / batch_size 3 = 4 batches
    
    // Check batch sizes
    assert_eq!(batches[0].len(), 3);
    assert_eq!(batches[1].len(), 3);
    assert_eq!(batches[2].len(), 3);
    assert_eq!(batches[3].len(), 1); // Last batch has remaining example
}

#[test]
fn test_dataloader_no_shuffle() {
    let dataset = create_test_dataset();
    let dataloader = DataLoader::new(&dataset, 2, false);
    
    let batches: Vec<_> = dataloader.collect();
    
    // Check that examples are in order when shuffle=false
    let first_batch = &batches[0];
    assert_eq!(first_batch.get(0).unwrap().data[0], 0.0);
    assert_eq!(first_batch.get(1).unwrap().data[0], 1.0);
    
    let second_batch = &batches[1];
    assert_eq!(second_batch.get(0).unwrap().data[0], 2.0);
    assert_eq!(second_batch.get(1).unwrap().data[0], 3.0);
}

#[test]
fn test_distributed_sampler() {
    let dataset = create_test_dataset();
    
    // Test with 2 replicas (processes)
    let sampler0 = DistributedSampler::new(&dataset, 2, 0);
    let sampler1 = DistributedSampler::new(&dataset, 2, 1);
    
    let indices0 = sampler0.indices();
    let indices1 = sampler1.indices();
    
    // Each sampler should get 5 examples
    assert_eq!(indices0.len(), 5);
    assert_eq!(indices1.len(), 5);
    
    // Check that indices don't overlap (except for padding)
    let mut all_indices = indices0.clone();
    all_indices.extend(&indices1);
    all_indices.sort();
    
    // The total should cover all original indices
    for i in 0..10 {
        assert!(all_indices.contains(&i));
    }
}

#[test]
fn test_distributed_sampler_padding() {
    let dataset = create_test_dataset();
    
    // Test with 3 replicas - 10 examples don't divide evenly by 3
    let sampler0 = DistributedSampler::new(&dataset, 3, 0);
    let sampler1 = DistributedSampler::new(&dataset, 3, 1);
    let sampler2 = DistributedSampler::new(&dataset, 3, 2);
    
    // Each should get 4 examples (ceil(10/3) = 4)
    assert_eq!(sampler0.indices().len(), 4);
    assert_eq!(sampler1.indices().len(), 4);
    assert_eq!(sampler2.indices().len(), 4);
    
    // Total padded size should be 12
    assert_eq!(sampler0.indices().len() + sampler1.indices().len() + sampler2.indices().len(), 12);
}

#[test]
fn test_example_field_access() {
    // Example only stores Vec<f32> now, not HashMap
    let example = Example { 
        data: vec![1.0, 2.0, 3.0]
    };
    
    assert_eq!(example.data[0], 1.0);
    assert_eq!(example.data[1], 2.0);
    assert_eq!(example.data[2], 3.0);
}