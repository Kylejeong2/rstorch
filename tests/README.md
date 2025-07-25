# RSTorch Test Suite

This directory contains comprehensive tests for the RSTorch deep learning library, organized into unit tests and integration tests.

## Test Organization

### Unit Tests (`unit/`)
Unit tests focus on testing individual components in isolation:

- **`test_tensor_unit.rs`**: Tests individual tensor operations, creation, and memory management
- **`test_nn_unit.rs`**: Tests neural network components (layers, activations, losses) independently  
- **`test_optim_unit.rs`**: Tests optimizers and parameter management in isolation
- **`autograd_test.rs`**: Tests individual autograd backward functions
- **`test_autograd.rs`**: Comprehensive autograd tests with numerical verification
- **`basic.rs`**: Basic functionality tests with mock functions
- **`test_operations.rs`**: Tests for tensor operations

### Integration Tests (`integration/`)
Integration tests verify that components work together correctly:

- **`test_full_training.rs`**: End-to-end training pipeline verification
- **`test_nn_components.rs`**: Tests neural network components working together
- **`test_nn.rs`**: Neural network integration tests
- **`test_dataset.rs`**: Dataset and data loading integration tests
- **`distributed_test.rs`**: Distributed training functionality tests
- **`test_distributed.rs`**: Comprehensive distributed training tests

## Running Tests

### Run All Tests
```bash
cargo test
```

### Run Unit Tests Only
```bash
cargo test --test "unit/*"
```

### Run Integration Tests Only  
```bash
cargo test --test "integration/*"
```

### Run Specific Test Files
```bash
# Run tensor unit tests
cargo test --test test_tensor_unit

# Run neural network integration tests
cargo test --test test_nn_components

# Run training pipeline tests
cargo test --test test_full_training
```

### Run Tests with Output
```bash
cargo test -- --nocapture
```

### Run Tests in Parallel
```bash
cargo test -j 4  # Use 4 threads
```

## Test Categories

### 🔧 **Unit Tests**
- ✅ Tensor creation and validation
- ✅ Individual tensor operations (add, mul, matmul, etc.)
- ✅ Activation functions (ReLU, Sigmoid, Softmax)
- ✅ Loss functions (MSE, CrossEntropy)  
- ✅ Neural network layers (Linear)
- ✅ Parameter management
- ✅ Optimizers (SGD with momentum)
- ✅ Autograd backward functions with numerical verification

### 🚀 **Integration Tests**
- ✅ Complete training workflows
- ✅ Multi-layer neural networks
- ✅ Model state persistence (save/load)
- ✅ Batch processing
- ✅ Parameter updates and gradient flow
- ✅ Data pipeline integration
- ✅ Distributed training (when available)

## Test Coverage

The test suite covers:

### Core Functionality
- [x] Tensor operations and memory safety
- [x] Neural network forward passes
- [x] Loss computation
- [x] Parameter management and updates
- [x] Optimizer functionality
- [x] Model serialization

### Edge Cases
- [x] Invalid input handling
- [x] Shape mismatches
- [x] Null pointer safety
- [x] Memory management
- [x] Error propagation

### Performance & Integration
- [x] Batch processing
- [x] Multi-layer networks
- [x] Training loops
- [x] Gradient computation verification

## Writing New Tests

### Unit Test Guidelines
1. Test one component at a time
2. Use descriptive test names: `test_component_behavior()`
3. Include error cases and edge conditions
4. Verify exact numerical results where possible
5. Test both success and failure paths

### Integration Test Guidelines
1. Test realistic workflows
2. Combine multiple components
3. Use realistic data sizes
4. Test end-to-end functionality
5. Include performance considerations

### Example Unit Test
```rust
#[test]
fn test_tensor_addition() {
    let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Failed to create tensor");
    let b = Tensor::from_vec(vec![3.0, 4.0], &[2]).expect("Failed to create tensor");
    
    let c = a.add(&b);
    assert_eq!(c.to_vec(), vec![4.0, 6.0], "Addition should be element-wise");
}
```

### Example Integration Test
```rust
#[test]
fn test_training_workflow() {
    let mut model = Linear::new(2, 1, true);
    let optimizer = SGD::new(model.parameters(), 0.01, 0.9);
    
    // Training loop
    for epoch in 0..10 {
        let prediction = model.forward(&[&input]);
        let loss = loss_fn.forward(&[&prediction, &target]);
        // ... backward pass and optimization
    }
    
    // Verify model learned something
    assert!(final_loss < initial_loss, "Model should improve over training");
}
```

## Troubleshooting

### Common Issues

1. **C++ Backend Not Available**: Some tests may fail if the C++ backend is not compiled
   - Solution: Ensure build.rs compiles successfully
   - Check that C++ compiler is available

2. **Memory Safety Issues**: Tests involving raw pointers may fail  
   - Check tensor creation and cleanup
   - Verify null pointer handling

3. **Numerical Precision**: Floating-point comparisons may fail
   - Use appropriate epsilon values for comparisons
   - Consider using `assert!((a - b).abs() < 1e-6)` instead of `assert_eq!`

4. **Distributed Tests**: May require MPI environment
   - Set appropriate environment variables
   - May need to skip on single-machine setups

### Test Debugging
```bash
# Run tests with backtrace
RUST_BACKTRACE=1 cargo test

# Run single test with output
cargo test test_specific_function -- --nocapture --exact

# Run tests with debug output
RUST_LOG=debug cargo test
```

## Contributing

When adding new functionality:

1. **Always add unit tests** for new components
2. **Add integration tests** for new workflows  
3. **Test error conditions** and edge cases
4. **Verify numerical correctness** with known results
5. **Document test expectations** in comments
6. **Update this README** if adding new test categories

## Continuous Integration

Tests should pass on:
- ✅ Linux (Ubuntu latest)
- ✅ macOS (latest)  
- ⚠️ Windows (may have C++ backend issues)

All tests are run automatically on pull requests and commits to main branch.