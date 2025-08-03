// Distributed neural network tests - Tests for distributed data parallel training functionality
// Tests distributed data parallel wrapper, process management, and distributed neural network operations
// Connected to: src/distributed/distributed.rs, src/nn/parallel.rs, src/nn/modules/
// Used by: Test suite verification of distributed neural network training

use rstorch::distributed::{get_rank, get_world_size, init_process_group_rs};
use rstorch::nn::{Module, DistributedDataParallel, Linear};
use rstorch::tensor::Tensor;
use std::env;

#[test]
fn test_get_rank_default() {
    // When environment variable is not set, should return 0
    env::remove_var("OMPI_COMM_WORLD_RANK");
    assert_eq!(get_rank(), Ok(0));
}

#[test]
fn test_get_rank_from_env() {
    env::set_var("OMPI_COMM_WORLD_RANK", "3");
    assert_eq!(get_rank(), Ok(3));
    env::remove_var("OMPI_COMM_WORLD_RANK");
}

#[test]
fn test_get_world_size_default() {
    // When environment variable is not set, should return 1
    env::remove_var("OMPI_COMM_WORLD_SIZE");
    assert_eq!(get_world_size(), Ok(1));
}

#[test]
fn test_get_world_size_from_env() {
    env::set_var("OMPI_COMM_WORLD_SIZE", "4");
    assert_eq!(get_world_size(), Ok(4));
    env::remove_var("OMPI_COMM_WORLD_SIZE");
}

#[test]
fn test_distributed_data_parallel_creation() {
    let linear = Linear::new(10, 5, true);
    let ddp = DistributedDataParallel::new(Box::new(linear));
    
    // Should be able to use it as a regular module
    let input = Tensor::zeros(&[2, 10]).unwrap();
    let result = ddp.forward(&[&input]);
    assert!(result.is_ok());
}

#[test]
fn test_distributed_data_parallel_forward() {
    let linear = Linear::new(3, 2, true);
    let ddp = DistributedDataParallel::new(Box::new(linear));
    
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let output = ddp.forward(&[&input]).unwrap();
    
    assert_eq!(output.shape(), &[2, 2]);
}

#[test]
fn test_distributed_data_parallel_training_mode() {
    let linear = Linear::new(5, 3, true);
    let mut ddp = DistributedDataParallel::new(Box::new(linear));
    
    // Test training mode
    ddp.train();
    assert!(ddp.is_training());
    
    // Test eval mode
    ddp.eval();
    assert!(!ddp.is_training());
}

#[test]
fn test_distributed_data_parallel_display() {
    let linear = Linear::new(4, 2, false);
    let ddp = DistributedDataParallel::new(Box::new(linear));
    
    let display_str = format!("{}", ddp);
    assert!(display_str.contains("DistributedDataParallel"));
    assert!(display_str.contains("Linear"));
}

#[cfg(test)]
mod distributed_integration {
    use super::*;
    
    #[test]
    #[ignore] // This test requires MPI to be set up
    fn test_init_process_group() {
        // This would only work in an actual distributed environment
        init_process_group_rs(0, 1).ok();
        assert_eq!(get_rank(), Ok(0));
        assert_eq!(get_world_size(), Ok(1));
    }
    
    #[test]
    #[ignore] // This test requires actual tensor broadcasting capability
    fn test_broadcast_parameters() {
        env::set_var("OMPI_COMM_WORLD_RANK", "0");
        env::set_var("OMPI_COMM_WORLD_SIZE", "2");
        
        let linear = Linear::new(10, 5, true);
        let _ddp = DistributedDataParallel::new(Box::new(linear));
        
        // In a real distributed setting, parameters would be broadcast from rank 0
        // to all other ranks during initialization
        
        env::remove_var("OMPI_COMM_WORLD_RANK");
        env::remove_var("OMPI_COMM_WORLD_SIZE");
    }
}