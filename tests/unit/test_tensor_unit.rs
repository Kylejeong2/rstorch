// Unit tests for Tensor operations - Tests individual tensor methods in isolation
// Verifies tensor creation, operations, and memory management work correctly
// Connected to: src/tensor.rs, src/csrc/*.h
// Used by: Unit test suite to verify tensor functionality

use rstorch::Tensor;

#[test]
fn test_tensor_creation() {
    // Test from_vec creation
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert!(tensor.is_ok(), "Tensor creation should succeed");
    
    let tensor = tensor.unwrap();
    assert_eq!(tensor.shape(), &[2, 2], "Tensor should have correct shape");
    assert_eq!(tensor.ndim, 2, "Tensor should have correct number of dimensions");
    assert_eq!(tensor.numel, 4, "Tensor should have correct number of elements");
    assert_eq!(tensor.device, "cpu", "Tensor should be on CPU by default");
}

#[test]
fn test_tensor_creation_with_device() {
    let tensor = Tensor::from_vec_device(vec![1.0, 2.0], &[2], "cpu");
    assert!(tensor.is_ok(), "Tensor creation with device should succeed");
    
    let tensor = tensor.unwrap();
    assert_eq!(tensor.device, "cpu", "Tensor should have correct device");
}

#[test]
fn test_tensor_zeros() {
    let zeros = Tensor::zeros(&[3, 2]);
    assert!(zeros.is_ok(), "Zeros tensor creation should succeed");
    
    let zeros = zeros.unwrap();
    assert_eq!(zeros.shape(), &[3, 2], "Zeros tensor should have correct shape");
    
    let data = zeros.to_vec();
    for &val in &data {
        assert_eq!(val, 0.0, "All values should be zero");
    }
}

#[test]
fn test_tensor_data_length_validation() {
    // Test mismatched data length
    let result = Tensor::from_vec(vec![1.0, 2.0], &[2, 2]); // 2 elements, but shape needs 4
    assert!(result.is_err(), "Should fail with mismatched data length");
}

#[test]
fn test_tensor_to_vec() {
    let original_data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_vec(original_data.clone(), &[2, 2]).expect("Tensor creation failed");
    
    let retrieved_data = tensor.to_vec();
    assert_eq!(retrieved_data, original_data, "Retrieved data should match original");
}

#[test]
fn test_tensor_clone() {
    let original = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Original tensor creation failed");
    let cloned = original.clone();
    
    assert_eq!(cloned.shape(), original.shape(), "Cloned tensor should have same shape");
    assert_eq!(cloned.to_vec(), original.to_vec(), "Cloned tensor should have same data");
    assert_eq!(cloned.device, original.device, "Cloned tensor should have same device");
}

#[test]
fn test_tensor_requires_grad() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Tensor creation failed");
    assert!(!tensor.requires_grad, "Tensor should not require grad by default");
    
    // Test with gradient requirement
    let mut tensor_with_grad = tensor.clone();
    tensor_with_grad.requires_grad = true;
    assert!(tensor_with_grad.requires_grad, "Tensor should require grad when set");
}

#[test]
fn test_tensor_addition() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Tensor a creation failed");
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("Tensor b creation failed");
    
    let c = a.add(&b);
    assert_eq!(c.shape(), &[2, 2], "Result should have correct shape");
    
    let result_data = c.to_vec();
    let expected = vec![6.0, 8.0, 10.0, 12.0];
    assert_eq!(result_data, expected, "Addition result should be correct");
}

#[test]
fn test_tensor_subtraction() {
    let a = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("Tensor a creation failed");
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Tensor b creation failed");
    
    let c = a.sub(&b);
    let result_data = c.to_vec();
    let expected = vec![4.0, 4.0, 4.0, 4.0];
    assert_eq!(result_data, expected, "Subtraction result should be correct");
}

#[test]
fn test_tensor_element_wise_multiplication() {
    let a = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).expect("Tensor a creation failed");
    let b = Tensor::from_vec(vec![2.0, 2.0, 2.0, 2.0], &[2, 2]).expect("Tensor b creation failed");
    
    let c = a.mul(&b);
    let result_data = c.to_vec();
    let expected = vec![4.0, 6.0, 8.0, 10.0];
    assert_eq!(result_data, expected, "Element-wise multiplication result should be correct");
}

#[test]
fn test_tensor_scalar_multiplication() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Tensor creation failed");
    
    let b = a.scalar_mul(2.5);
    let result_data = b.to_vec();
    let expected = vec![2.5, 5.0, 7.5, 10.0];
    assert_eq!(result_data, expected, "Scalar multiplication result should be correct");
}

#[test]  
fn test_tensor_matrix_multiplication() {
    // Test 2x2 @ 2x2 = 2x2
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Tensor a creation failed");
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("Tensor b creation failed");
    
    let c = a.matmul(&b);
    assert_eq!(c.shape(), &[2, 2], "Matrix multiplication result should have correct shape");
    
    // Expected result: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let result_data = c.to_vec();
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    assert_eq!(result_data, expected, "Matrix multiplication result should be correct");
}

#[test]
fn test_tensor_transpose() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("Tensor creation failed");
    
    let b = a.transpose();
    assert_eq!(b.shape(), &[3, 2], "Transposed tensor should have swapped dimensions");
    
    let result_data = b.to_vec();
    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // Column-major order
    assert_eq!(result_data, expected, "Transpose result should be correct");
}

#[test]
fn test_tensor_sum() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Tensor creation failed");
    
    // Sum all elements
    let sum_all = a.sum(-1, false);
    assert_eq!(sum_all.shape(), &[1], "Sum all should result in scalar");
    assert_eq!(sum_all.to_vec()[0], 10.0, "Sum of all elements should be 10");
    
    // Sum along axis 0
    let sum_axis0 = a.sum(0, false);
    assert_eq!(sum_axis0.shape(), &[2], "Sum along axis 0 should have shape [2]");
    let sum_data = sum_axis0.to_vec();
    assert_eq!(sum_data, vec![4.0, 6.0], "Sum along axis 0 should be [4, 6]");
}

#[test]
fn test_tensor_activations() {
    let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], &[4]).expect("Tensor creation failed");
    
    // Test ReLU
    let relu_result = a.relu();
    let relu_data = relu_result.to_vec();
    assert_eq!(relu_data, vec![0.0, 0.0, 1.0, 2.0], "ReLU should zero negative values");
    
    // Test Sigmoid
    let sigmoid_result = a.sigmoid();
    let sigmoid_data = sigmoid_result.to_vec();
    for &val in &sigmoid_data {
        assert!(val > 0.0 && val < 1.0, "Sigmoid output {} should be between 0 and 1", val);
    }
    
    // Test specific sigmoid values
    let zero_input = Tensor::from_vec(vec![0.0], &[1]).expect("Zero tensor creation failed");
    let sigmoid_zero = zero_input.sigmoid();
    let sigmoid_zero_val = sigmoid_zero.to_vec()[0];
    assert!((sigmoid_zero_val - 0.5).abs() < 1e-6, "Sigmoid(0) should be 0.5, got {}", sigmoid_zero_val);
}

#[test]
fn test_tensor_log() {
    let a = Tensor::from_vec(vec![1.0, 2.718281828, 7.389], &[3]).expect("Tensor creation failed");
    
    let log_result = a.log();
    let log_data = log_result.to_vec();
    
    // ln(1) = 0, ln(e) ≈ 1, ln(e^2) ≈ 2
    assert!((log_data[0] - 0.0).abs() < 1e-6, "ln(1) should be 0");
    assert!((log_data[1] - 1.0).abs() < 1e-4, "ln(e) should be 1");
    assert!((log_data[2] - 2.0).abs() < 1e-3, "ln(e^2) should be 2");
}

#[test]
fn test_tensor_softmax() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Tensor creation failed");
    
    let softmax_result = a.softmax(-1);
    let softmax_data = softmax_result.to_vec();
    
    // Check that outputs sum to 1
    let sum: f32 = softmax_data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Softmax outputs should sum to 1, got {}", sum);
    
    // Check that all outputs are positive
    for &val in &softmax_data {
        assert!(val > 0.0, "Softmax output {} should be positive", val);
    }
    
    // Check that larger inputs produce larger outputs
    assert!(softmax_data[2] > softmax_data[1], "Softmax should preserve ordering");
    assert!(softmax_data[1] > softmax_data[0], "Softmax should preserve ordering");
}

#[test]
fn test_tensor_zeros_like_ones_like() {
    let original = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Original tensor creation failed");
    
    // Test zeros_like
    let zeros = original.zeros_like();
    assert!(zeros.is_ok(), "zeros_like should succeed");
    let zeros = zeros.unwrap();
    
    assert_eq!(zeros.shape(), original.shape(), "zeros_like should have same shape");
    let zeros_data = zeros.to_vec();
    for &val in &zeros_data {
        assert_eq!(val, 0.0, "zeros_like should contain only zeros");
    }
    
    // Test ones_like
    let ones = original.ones_like();
    assert!(ones.is_ok(), "ones_like should succeed");
    let ones = ones.unwrap();
    
    assert_eq!(ones.shape(), original.shape(), "ones_like should have same shape");
    let ones_data = ones.to_vec();
    for &val in &ones_data {
        assert_eq!(val, 1.0, "ones_like should contain only ones");
    }
}

#[test]
fn test_tensor_gradient_management() {
    let mut tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Tensor creation failed");
    
    // Test initial gradient state
    assert!(tensor.grad.is_none(), "Tensor should have no gradient initially");
    
    // Test zero_grad
    tensor.zero_grad();
    assert!(tensor.grad.is_none(), "zero_grad should clear gradients");
    
    // Test requires_grad
    tensor.requires_grad = true;
    assert!(tensor.requires_grad, "requires_grad should be settable");
}

#[test]
fn test_tensor_device_property() {
    let tensor = Tensor::from_vec_device(vec![1.0, 2.0], &[2], "cpu").expect("Tensor creation failed");
    assert_eq!(tensor.device, "cpu", "Tensor should have correct device");
    
    // Test default device
    let default_tensor = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Default tensor creation failed");
    assert_eq!(default_tensor.device, "cpu", "Default device should be CPU");
}

#[test]
fn test_tensor_error_cases() {
    // Test invalid matrix multiplication shapes
    let a = Tensor::from_vec(vec![1.0, 2.0], &[1, 2]).expect("Tensor a creation failed");
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("Tensor b creation failed");
    
    // This should panic due to incompatible shapes - using AssertUnwindSafe
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        a.matmul(&b);
    }));
    assert!(result.is_err(), "Matrix multiplication with incompatible shapes should panic");
    
    // Test transpose on non-2D tensor should panic
    let tensor_1d = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("1D tensor creation failed");
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tensor_1d.transpose();
    }));
    assert!(result.is_err(), "Transpose on 1D tensor should panic");
}