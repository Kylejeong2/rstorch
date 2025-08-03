// Tensor operations tests - Tests for tensor creation, manipulation, and arithmetic operations
// Tests comprehensive tensor operations including creation, reshaping, arithmetic, and device handling
// Connected to: src/tensor.rs, src/csrc/
// Used by: Test suite verification of tensor operation correctness

use rstorch::tensor::Tensor;

#[test]
fn test_tensor_creation() {
    let tensor = Tensor::new(Some(vec![1.0, 2.0, 3.0]), "cpu", false).unwrap();
    assert_eq!(tensor.shape(), &[3]);
    assert_eq!(tensor.ndim, 1);
    assert_eq!(tensor.device, "cpu");
    assert_eq!(tensor.numel, 3);
    assert!(!tensor.requires_grad);
}

#[test]
fn test_tensor_from_vec() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data.clone(), &[2, 3]).unwrap();
    
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.ndim, 2);
    assert_eq!(tensor.to_vec(), data);
}

#[test]
fn test_tensor_zeros() {
    let tensor = Tensor::zeros(&[3, 4]).unwrap();
    assert_eq!(tensor.shape(), &[3, 4]);
    assert_eq!(tensor.numel, 12);
    
    let data = tensor.to_vec();
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_tensor_ones_like() {
    let original = Tensor::zeros(&[2, 3]).unwrap();
    let ones = original.ones_like().unwrap();
    
    assert_eq!(ones.shape(), original.shape());
    let data = ones.to_vec();
    assert!(data.iter().all(|&x| x == 1.0));
}

#[test]
fn test_tensor_zeros_like() {
    let original = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let zeros = original.zeros_like().unwrap();
    
    assert_eq!(zeros.shape(), original.shape());
    let data = zeros.to_vec();
    assert!(data.iter().all(|&x| x == 0.0));
}

#[test]
fn test_tensor_add() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
    
    let c = a.add(&b);
    assert_eq!(c.to_vec(), vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_tensor_sub() {
    let a = Tensor::from_vec(vec![5.0, 7.0, 9.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    
    let c = a.sub(&b);
    assert_eq!(c.to_vec(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn test_tensor_mul() {
    let a = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0], &[3]).unwrap();
    
    let c = a.mul(&b);
    assert_eq!(c.to_vec(), vec![10.0, 18.0, 28.0]);
}

#[test]
fn test_tensor_scalar_mul() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = a.scalar_mul(2.5);
    
    assert_eq!(b.to_vec(), vec![2.5, 5.0, 7.5]);
}

#[test]
fn test_tensor_matmul() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
    
    let c = a.matmul(&b);
    assert_eq!(c.shape(), &[2, 2]);
    // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_tensor_sum() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    
    // Sum all elements
    let sum_all = a.sum(-1, false);
    assert_eq!(sum_all.shape(), &[1]);
    assert_eq!(sum_all.to_vec(), vec![21.0]);
    
    // Sum along axis 0
    let sum_axis0 = a.sum(0, false);
    assert_eq!(sum_axis0.shape(), &[3]);
    assert_eq!(sum_axis0.to_vec(), vec![5.0, 7.0, 9.0]);
    
    // Sum along axis 1
    let sum_axis1 = a.sum(1, false);
    assert_eq!(sum_axis1.shape(), &[2]);
    assert_eq!(sum_axis1.to_vec(), vec![6.0, 15.0]);
    
    // Sum with keepdim
    let sum_keepdim = a.sum(1, true);
    assert_eq!(sum_keepdim.shape(), &[2, 1]);
}

#[test]
fn test_tensor_sigmoid() {
    let a = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]).unwrap();
    let sig = a.sigmoid();
    
    let values = sig.to_vec();
    assert!((values[0] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
    assert!(values[1] > 0.5 && values[1] < 1.0); // sigmoid(1) ≈ 0.731
    assert!(values[2] > 0.0 && values[2] < 0.5); // sigmoid(-1) ≈ 0.269
}

#[test]
fn test_tensor_relu() {
    let a = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
    let relu = a.relu();
    
    assert_eq!(relu.to_vec(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_tensor_log() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let log = a.log();
    
    let values = log.to_vec();
    assert!((values[0] - 0.0).abs() < 1e-6); // log(1) = 0
    assert!((values[1] - 0.693147).abs() < 1e-4); // log(2) ≈ 0.693
    assert!((values[2] - 1.098612).abs() < 1e-4); // log(3) ≈ 1.099
}

#[test]
fn test_tensor_softmax() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let softmax = a.softmax(-1);
    
    let values = softmax.to_vec();
    let sum: f32 = values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6); // Softmax should sum to 1
    
    // Check ordering: softmax preserves order
    assert!(values[0] < values[1]);
    assert!(values[1] < values[2]);
}

#[test]
fn test_tensor_clone() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = a.clone();
    
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.to_vec(), b.to_vec());
    assert_eq!(a.requires_grad, b.requires_grad);
}

#[test]
fn test_tensor_operators() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
    
    // Test Add operator
    let c = &a + &b;
    assert_eq!(c.to_vec(), vec![5.0, 7.0, 9.0]);
    
    // Test Mul operator with scalar
    let d = &a * 2.0;
    assert_eq!(d.to_vec(), vec![2.0, 4.0, 6.0]);
    
    // Test Mul operator with tensor
    let e = &a * &b;
    assert_eq!(e.to_vec(), vec![4.0, 10.0, 18.0]);
    
    // Test Div operator
    let f = a / 2.0;
    assert_eq!(f.to_vec(), vec![0.5, 1.0, 1.5]);
}

#[test]
fn test_tensor_grad_tracking() {
    let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    a.requires_grad = true;
    
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
    
    // Operations should track gradients
    let c = a.add(&b);
    assert!(c.requires_grad);
    
    let d = a.mul(&b);
    assert!(d.requires_grad);
}

#[test]
fn test_tensor_zero_grad() {
    let mut a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    a.requires_grad = true;
    
    // Set some gradient
    a.grad = Some(Box::new(Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap()));
    assert!(a.grad.is_some());
    
    // Zero the gradient
    a.zero_grad();
    assert!(a.grad.is_none());
}

#[test] 
fn test_tensor_shape_operations() {
    // Test various shape-related operations
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq!(a.ndim, 2);
    assert_eq!(a.numel, 6);
    
    // Test with different shapes
    let b = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 4]).unwrap();
    assert_eq!(b.shape(), &[1, 1, 4]);
    assert_eq!(b.ndim, 3);
    assert_eq!(b.numel, 4);
}