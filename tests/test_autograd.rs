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