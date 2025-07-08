use rstorch::autograd::*;
use ndarray::ArrayD;

#[test]
fn test_add_backward() {
    let grad = ArrayD::from_elem(vec![2, 2], 1.0);
    let add_bw = AddBackward;
    let grads = add_bw.backward(&grad);
    
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0], grad);
    assert_eq!(grads[1], grad);
}

#[test]
fn test_add_broadcasted_backward() {
    let grad = ArrayD::from_elem(vec![2, 2], 1.0);
    let add_bw = AddBroadcastedBackward {
        x_shape: vec![2, 2],
        y_shape: vec![1, 2],
    };
    let grads = add_bw.backward(&grad);
    
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), &[2, 2]);
    assert_eq!(grads[1].shape(), &[1, 2]);
}

#[test]
fn test_scalar_mul_backward() {
    let grad = ArrayD::from_elem(vec![2, 2], 2.0);
    let scalar_mul_bw = ScalarMulBackward { scalar: 3.0 };
    let grads = scalar_mul_bw.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0], grad * 3.0);
}

#[test]
fn test_elementwise_mul_backward() {
    let x = ArrayD::from_elem(vec![2, 2], 2.0);
    let y = ArrayD::from_elem(vec![2, 2], 3.0);
    let grad = ArrayD::from_elem(vec![2, 2], 1.0);
    
    let mul_bw = ElementwiseMulBackward { x: x.clone(), y: y.clone() };
    let grads = mul_bw.backward(&grad);
    
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0], &y * &grad);
    assert_eq!(grads[1], &x * &grad);
}

#[test]
fn test_sum_backward() {
    let grad = ArrayD::from_elem(vec![], 1.0); // scalar gradient
    let sum_bw = SumBackward {
        input_shape: vec![2, 2],
        axis: None,
        keepdim: false,
    };
    let grads = sum_bw.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
    assert_eq!(grads[0], ArrayD::from_elem(vec![2, 2], 1.0));
}

#[test]
fn test_reshape_backward() {
    let grad = ArrayD::from_elem(vec![4], 1.0);
    let reshape_bw = ReshapeBackward {
        input_shape: vec![2, 2],
    };
    let grads = reshape_bw.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
}

#[test]
fn test_sigmoid_backward() {
    let input = ArrayD::from_elem(vec![2, 2], 0.5);
    let grad = ArrayD::from_elem(vec![2, 2], 1.0);
    
    let sigmoid_bw = SigmoidBackward { input: input.clone() };
    let grads = sigmoid_bw.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
    // Sigmoid derivative at 0.5 should be sigmoid(0.5) * (1 - sigmoid(0.5))
    let sigmoid_val = 1.0 / (1.0 + (-0.5_f32).exp());
    let expected_grad = sigmoid_val * (1.0 - sigmoid_val);
    assert!((grads[0][[0, 0]] - expected_grad).abs() < 1e-6);
} 