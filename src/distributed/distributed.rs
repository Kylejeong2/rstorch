use std::env;
use std::os::raw::c_int;

use crate::Tensor;

extern "C" {
    fn init_process_group(rank: c_int, world_size: c_int);
    fn broadcast_tensor(tensor: *mut std::ffi::c_void, src: c_int);
    fn allreduce_sum_tensor(tensor: *mut std::ffi::c_void);
}

pub fn init_process_group_rs(rank: usize, world_size: usize) {
    unsafe {
        init_process_group(rank as c_int, world_size as c_int);
    }
}

/// Return the integer rank of the current process.
pub fn get_rank() -> usize {
    env::var("OMPI_COMM_WORLD_RANK")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Return total number of processes in the group.
pub fn get_world_size() -> usize {
    env::var("OMPI_COMM_WORLD_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
}

pub fn broadcast_tensor_rs(tensor: &mut Tensor, src: usize) {
    unsafe {
        broadcast_tensor(tensor.tensor as *mut std::ffi::c_void, src as c_int);
    }
}

pub fn allreduce_sum_tensor_rs(tensor: &mut Tensor) {
    unsafe {
        allreduce_sum_tensor(tensor.tensor as *mut std::ffi::c_void);
    }
}