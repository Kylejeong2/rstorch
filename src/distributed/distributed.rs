// Distributed training support - Rust bindings for distributed tensor operations
// Provides process group initialization, broadcast, and allreduce operations using NCCL backend
// Connected to: src/csrc/distributed.h, src/csrc/distributed.cpp, src/tensor.rs
// Used by: src/distributed/run/mod.rs, tests/distributed_test.rs, tests/test_distributed.rs

use std::env;
use std::os::raw::c_int;

use crate::{Tensor, CTensor};

extern "C" {
    fn init_process_group(backend: *const std::os::raw::c_char, rank: c_int, world_size: c_int);
    fn broadcast_tensor(tensor: *mut CTensor, src: c_int);
    fn allreduce_sum_tensor(tensor: *mut CTensor);
}

pub fn init_process_group_rs(rank: usize, world_size: usize) -> Result<(), String> {
    let backend = std::ffi::CString::new("nccl")
        .map_err(|_| "Failed to create CString for backend")?;
    unsafe {
        init_process_group(backend.as_ptr(), rank as c_int, world_size as c_int);
    }
    Ok(())
}

/// Return the integer rank of the current process.
pub fn get_rank() -> Result<usize, String> {
    let rank = match env::var("OMPI_COMM_WORLD_RANK") {
        Ok(rank_str) => {
            rank_str.parse::<usize>()
                .unwrap_or(0) // Return default rank 0 if parsing fails
        }
        Err(_) => 0 // Return default rank 0 if env var not set
    };
    Ok(rank)
}

/// Return total number of processes in the group.
pub fn get_world_size() -> Result<usize, String> {
    let world_size = match env::var("OMPI_COMM_WORLD_SIZE") {
        Ok(size_str) => {
            size_str.parse::<usize>()
                .unwrap_or(1) // Return default world size 1 if parsing fails
        }
        Err(_) => 1 // Return default world size 1 if env var not set
    };
    Ok(world_size)
}

pub fn broadcast_tensor_rs(tensor: &mut Tensor, src: usize) -> Result<(), String> {
    if tensor.tensor.is_null() {
        return Err("Cannot broadcast null tensor".to_string());
    }
    unsafe {
        broadcast_tensor(tensor.tensor, src as c_int);
    }
    Ok(())
}

pub fn allreduce_sum_tensor_rs(tensor: &mut Tensor) -> Result<(), String> {
    if tensor.tensor.is_null() {
        return Err("Cannot allreduce null tensor".to_string());
    }
    unsafe {
        allreduce_sum_tensor(tensor.tensor);
    }
    Ok(())
}