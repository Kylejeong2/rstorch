// Distributed module - Exports distributed training functionality
// Aggregates distributed operations and run utilities for multi-process training
// Connected to: src/distributed/distributed.rs, src/distributed/run/mod.rs
// Used by: src/lib.rs, examples, tests

pub mod run;
pub mod distributed;
pub use distributed::*;