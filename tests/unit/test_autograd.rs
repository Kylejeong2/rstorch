// Comprehensive autograd tests - Extended test suite for automatic differentiation
// Tests backward pass implementations for all supported operations with detailed gradient verification
// Connected to: src/autograd/functions.rs
// Used by: Test suite comprehensive verification of gradient computation accuracy

use rstorch::autograd::functions::*;
use ndarray::{ArrayD, IxDyn};

#[test]
fn test_add_backward() {
    let grad = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0);
    let backward = AddBackward;
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0], grad);
    assert_eq!(grads[1], grad);
}

#[test]
fn test_add_broadcasted_backward() {
    let grad = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0);
    let backward = AddBroadcastedBackward {
        x_shape: vec![2, 3],
        y_shape: vec![1, 3],
    };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), &[2, 3]);
    assert_eq!(grads[1].shape(), &[1, 3]);
}

#[test]
fn test_sub_backward() {
    let grad = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0);
    let backward = SubBackward;
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0], grad);
    assert_eq!(grads[1], -&grad);
}

#[test]
fn test_scalar_mul_backward() {
    let grad = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0);
    let backward = ScalarMulBackward { scalar: 2.5 };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0], grad * 2.5);
}

#[test]
fn test_elementwise_mul_backward() {
    let x = ArrayD::from_elem(IxDyn(&[2, 3]), 2.0);
    let y = ArrayD::from_elem(IxDyn(&[2, 3]), 3.0);
    let grad = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0);
    
    let backward = ElementwiseMulBackward { x: x.clone(), y: y.clone() };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0], &y * &grad);
    assert_eq!(grads[1], &x * &grad);
}

#[test]
fn test_matmul_backward() {
    let x = ArrayD::from_elem(IxDyn(&[2, 3]), 1.0);
    let y = ArrayD::from_elem(IxDyn(&[3, 4]), 1.0);
    let grad = ArrayD::from_elem(IxDyn(&[2, 4]), 1.0);
    
    let backward = MatmulBackward { x: x.clone(), y: y.clone() };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), &[2, 3]);
    assert_eq!(grads[1].shape(), &[3, 4]);
}

#[test]
fn test_sum_backward() {
    let grad = ArrayD::from_elem(IxDyn(&[1]), 2.0);
    let backward = SumBackward {
        input_shape: vec![2, 3],
        axis: None,
        keepdim: false,
    };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 3]);
    assert_eq!(grads[0], ArrayD::from_elem(IxDyn(&[2, 3]), 2.0));
}

#[test]
fn test_reshape_backward() {
    let grad = ArrayD::from_elem(IxDyn(&[6]), 1.0);
    let backward = ReshapeBackward {
        input_shape: vec![2, 3],
    };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 3]);
}

#[test]
fn test_sigmoid_backward() {
    let input = ArrayD::from_elem(IxDyn(&[2, 2]), 0.0);
    let grad = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0);
    
    let backward = SigmoidBackward { input };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
    // sigmoid(0) = 0.5, sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
    assert!((grads[0][[0, 0]] - 0.25).abs() < 1e-6);
}

#[test]
fn test_log_backward() {
    let x = ArrayD::from_elem(IxDyn(&[2, 2]), 2.0);
    let grad = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0);
    
    let backward = LogBackward { x: x.clone() };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0], grad / x);
}

#[test]
fn test_max_backward() {
    let input = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 3.0, 2.0, 4.0, 2.0, 5.0]).unwrap();
    let grad = ArrayD::from_elem(IxDyn(&[2]), 1.0);
    
    let backward = MaxBackward {
        input: input.clone(),
        axis: Some(1),
        keepdim: false,
    };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 3]);
}

#[test]
fn test_cross_entropy_loss_backward() {
    let logits = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).unwrap();
    let targets = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let grad = ArrayD::from_elem(IxDyn(&[2]), 1.0);
    
    let backward = CrossEntropyLossBackward {
        logits: logits.clone(),
        targets: targets.clone(),
    };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), &[2, 3]); // grad for logits
    assert_eq!(grads[1].shape(), &[2, 3]); // grad for targets (should be zeros)
}

#[test]
fn test_sin_backward() {
    let x = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 1.5708, 3.1416, 4.7124]).unwrap(); // 0, π/2, π, 3π/2
    let grad = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0);
    
    let backward = SinBackward { x: x.clone() };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
    
    // Check gradients numerically: d/dx sin(x) = cos(x)
    let expected_cos_values = x.mapv(|val| val.cos());
    let computed_grads = &grads[0];
    
    for i in 0..2 {
        for j in 0..2 {
            let expected = expected_cos_values[[i, j]];
            let computed = computed_grads[[i, j]];
            assert!((expected - computed).abs() < 1e-6, 
                   "SinBackward gradient mismatch at [{}, {}]: expected {}, got {}", i, j, expected, computed);
        }
    }
}

#[test]
fn test_cos_backward() {
    let x = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.0, 1.5708, 3.1416, 4.7124]).unwrap(); // 0, π/2, π, 3π/2
    let grad = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0);
    
    let backward = CosBackward { x: x.clone() };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
    
    // Check gradients numerically: d/dx cos(x) = -sin(x)
    let expected_neg_sin_values = x.mapv(|val| -val.sin());
    let computed_grads = &grads[0];
    
    for i in 0..2 {
        for j in 0..2 {
            let expected = expected_neg_sin_values[[i, j]];
            let computed = computed_grads[[i, j]];
            assert!((expected - computed).abs() < 1e-6, 
                   "CosBackward gradient mismatch at [{}, {}]: expected {}, got {}", i, j, expected, computed);
        }
    }
}

#[test]
fn test_log_backward() {
    let x = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let grad = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0);
    
    let backward = LogBackward { x: x.clone() };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
    
    // Check gradients numerically: d/dx ln(x) = 1/x
    let expected_grads = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.5, 1.0/3.0, 0.25]).unwrap();
    let computed_grads = &grads[0];
    
    for i in 0..2 {
        for j in 0..2 {
            let expected = expected_grads[[i, j]];
            let computed = computed_grads[[i, j]];
            assert!((expected - computed).abs() < 1e-6, 
                   "LogBackward gradient mismatch at [{}, {}]: expected {}, got {}", i, j, expected, computed);
        }
    }
}

#[test]
fn test_pow_backward() {
    let base = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0, 3.0, 4.0, 5.0]).unwrap();
    let exponent = ArrayD::from_elem(IxDyn(&[2, 2]), 2.0); // Square all values
    let grad = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0);
    
    let backward = PowBackward { 
        base: base.clone(), 
        exponent: exponent.clone() 
    };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), &[2, 2]); // grad for base
    assert_eq!(grads[1].shape(), &[2, 2]); // grad for exponent
    
    // Check base gradient numerically: d/dx (x^n) = n * x^(n-1)
    // For x^2: gradient = 2*x
    let expected_base_grads = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![4.0, 6.0, 8.0, 10.0]).unwrap();
    let computed_base_grads = &grads[0];
    
    for i in 0..2 {
        for j in 0..2 {
            let expected = expected_base_grads[[i, j]];
            let computed = computed_base_grads[[i, j]];
            assert!((expected - computed).abs() < 1e-6, 
                   "PowBackward base gradient mismatch at [{}, {}]: expected {}, got {}", i, j, expected, computed);
        }
    }
}

#[test]
fn test_sigmoid_backward() {
    let x = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
    let grad = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0);
    
    let backward = SigmoidBackward { x: x.clone() };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 1);
    assert_eq!(grads[0].shape(), &[2, 2]);
    
    // Check gradients numerically: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    let sigmoid_vals = x.mapv(|val| 1.0 / (1.0 + (-val).exp()));
    let expected_grads = &sigmoid_vals * sigmoid_vals.mapv(|val| 1.0 - val);
    let computed_grads = &grads[0];
    
    for i in 0..2 {
        for j in 0..2 {
            let expected = expected_grads[[i, j]];
            let computed = computed_grads[[i, j]];
            assert!((expected - computed).abs() < 1e-6, 
                   "SigmoidBackward gradient mismatch at [{}, {}]: expected {}, got {}", i, j, expected, computed);
        }
    }
}

#[test]
fn test_division_backward() {
    let x = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![4.0, 6.0, 8.0, 10.0]).unwrap();
    let y = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![2.0, 3.0, 4.0, 5.0]).unwrap();
    let grad = ArrayD::from_elem(IxDyn(&[2, 2]), 1.0);
    
    let backward = DivisionBackward { 
        x: x.clone(), 
        y: y.clone() 
    };
    let grads = backward.backward(&grad);
    
    assert_eq!(grads.len(), 2);
    assert_eq!(grads[0].shape(), &[2, 2]); // grad for x
    assert_eq!(grads[1].shape(), &[2, 2]); // grad for y
    
    // Check x gradient numerically: d/dx (x/y) = 1/y
    let expected_x_grads = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![0.5, 1.0/3.0, 0.25, 0.2]).unwrap();
    let computed_x_grads = &grads[0];
    
    for i in 0..2 {
        for j in 0..2 {
            let expected = expected_x_grads[[i, j]];
            let computed = computed_x_grads[[i, j]];
            assert!((expected - computed).abs() < 1e-6, 
                   "DivisionBackward x gradient mismatch at [{}, {}]: expected {}, got {}", i, j, expected, computed);
        }
    }
    
    // Check y gradient numerically: d/dy (x/y) = -x/y^2
    let expected_y_grads = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![-1.0, -6.0/9.0, -8.0/16.0, -10.0/25.0]).unwrap();
    let computed_y_grads = &grads[1];
    
    for i in 0..2 {
        for j in 0..2 {
            let expected = expected_y_grads[[i, j]];
            let computed = computed_y_grads[[i, j]];
            assert!((expected - computed).abs() < 1e-6, 
                   "DivisionBackward y gradient mismatch at [{}, {}]: expected {}, got {}", i, j, expected, computed);
        }
    }
}