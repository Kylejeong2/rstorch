use std::env;

use crate::Tensor;

/// Initialise the (fake) process group. In real code this would call into MPI / NCCL.
pub fn init_process_group(rank: usize, world_size: usize) {
    env::set_var("OMPI_COMM_WORLD_RANK", rank.to_string());
    env::set_var("OMPI_COMM_WORLD_SIZE", world_size.to_string());
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

/// Broadcast `tensor` from process `src` to all others.
///
/// This stub assumes single-process execution; it simply returns.
pub fn broadcast_tensor(_tensor: &mut Tensor, _src: usize) {
    // Real implementation would use MPI_Bcast / NCCL etc.
}

/// In-place all-reduce (sum) over `tensor` across all processes.
///
/// Stub = no-op for single process.
pub fn allreduce_sum_tensor(_tensor: &mut Tensor) {
    // Real implementation would call MPI_Allreduce etc.
}