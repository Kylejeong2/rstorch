# RStorch

Pytorch written in Rust 

Recreating PyTorch from scratch (C/C++, CUDA and Python, with multi-GPU support and automatic differentiation!)

## Testing

Run the unit-test suite with Cargo:

```bash
cargo test
```

To see println! output while tests execute:

```bash
cargo test -- --nocapture | cat
```
