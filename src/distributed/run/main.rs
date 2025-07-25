// Distributed training launcher - Main entry point for distributed script execution
// Simple wrapper that calls the distributed run functionality to spawn multi-process training
// Connected to: src/distributed/run/mod.rs
// Used by: Standalone binary for distributed training execution

/**
import sys
from . import run

def main():
    run.main()

if __name__ == "__main__":
    main()
*/

use rstorch::distributed::run;

fn main() {
    run::run().unwrap();
}