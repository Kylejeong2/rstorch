use rstorch::nn::{Module, Linear, Sigmoid, Softmax, ReLU, MSELoss, CrossEntropyLoss, Loss};
use rstorch::nn::functional;
use rstorch::tensor::Tensor;

#[test]
fn test_linear_layer() {
    let linear = Linear::new(3, 2, true);
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
    
    let output = linear.forward(&[&input]).unwrap();
    assert_eq!(output.shape(), &[1, 2]);
}

#[test]
fn test_linear_layer_batch() {
    let linear = Linear::new(4, 3, true);
    let input = Tensor::from_vec(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4]
    );
    
    let output = linear.forward(&[&input]).unwrap();
    assert_eq!(output.shape(), &[2, 3]);
}

#[test]
fn test_linear_layer_no_bias() {
    let linear = Linear::new(3, 2, false);
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]);
    
    let output = linear.forward(&[&input]).unwrap();
    assert_eq!(output.shape(), &[1, 2]);
}

#[test]
fn test_sigmoid_activation() {
    let sigmoid = Sigmoid::new();
    let input = Tensor::from_vec(vec![0.0, 1.0, -1.0, 2.0], &[2, 2]);
    
    let output = sigmoid.forward(&[&input]).unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    
    // Check that values are between 0 and 1
    let values = output.to_vec();
    for val in values {
        assert!(val >= 0.0 && val <= 1.0);
    }
}

#[test]
fn test_relu_activation() {
    let relu = ReLU::new();
    let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    
    let output = relu.forward(&[&input]).unwrap();
    assert_eq!(output.shape(), &[5]);
    
    // Check that negative values become 0
    let values = output.to_vec();
    assert_eq!(values[0], 0.0);
    assert_eq!(values[1], 0.0);
    assert_eq!(values[2], 0.0);
    assert!(values[3] > 0.0);
    assert!(values[4] > 0.0);
}

#[test]
fn test_softmax_activation() {
    let softmax = Softmax::new(-1);
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
    
    let output = softmax.forward(&[&input]).unwrap();
    assert_eq!(output.shape(), &[3]);
    
    // Check that values sum to 1
    let values = output.to_vec();
    let sum: f32 = values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_mse_loss() {
    let loss_fn = MSELoss::new();
    let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let targets = Tensor::from_vec(vec![1.5, 2.5, 3.5, 4.5], &[4]);
    
    let loss = loss_fn.loss(&predictions, &targets);
    assert_eq!(loss.shape(), &[1]);
    
    // MSE should be ((0.5)^2 * 4) / 4 = 0.25
    let loss_val = loss.to_vec()[0];
    assert!((loss_val - 0.25).abs() < 1e-6);
}

#[test]
fn test_cross_entropy_loss() {
    let loss_fn = CrossEntropyLoss::new();
    let logits = Tensor::from_vec(vec![2.0, 1.0, 0.1, 3.0, 1.0, 0.1], &[2, 3]);
    let targets = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]);
    
    let loss = loss_fn.loss(&logits, &targets);
    assert_eq!(loss.shape(), &[1]);
}

#[test]
fn test_functional_sigmoid() {
    let input = Tensor::from_vec(vec![0.0, 1.0, -1.0], &[3]);
    let output = functional::sigmoid(&input);
    
    assert_eq!(output.shape(), &[3]);
    let values = output.to_vec();
    assert!((values[0] - 0.5).abs() < 1e-6); // sigmoid(0) = 0.5
}

#[test]
fn test_functional_relu() {
    let input = Tensor::from_vec(vec![-1.0, 0.0, 1.0], &[3]);
    let output = functional::relu(&input);
    
    assert_eq!(output.shape(), &[3]);
    let values = output.to_vec();
    assert_eq!(values[0], 0.0);
    assert_eq!(values[1], 0.0);
    assert_eq!(values[2], 1.0);
}

#[test]
fn test_functional_softmax() {
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
    let output = functional::softmax(&input, -1);
    
    assert_eq!(output.shape(), &[3]);
    let values = output.to_vec();
    let sum: f32 = values.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_functional_log_softmax() {
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
    let output = functional::log_softmax(&input, -1);
    
    assert_eq!(output.shape(), &[3]);
    // All log probabilities should be negative
    let values = output.to_vec();
    for val in values {
        assert!(val < 0.0);
    }
}

#[test]
fn test_functional_mse_loss() {
    let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
    let targets = Tensor::from_vec(vec![1.5, 2.5, 3.5], &[3]);
    
    let loss = functional::mse_loss(&predictions, &targets);
    assert_eq!(loss.shape(), &[1]);
    
    // MSE should be ((0.5)^2 * 3) / 3 = 0.25
    let loss_val = loss.to_vec()[0];
    assert!((loss_val - 0.25).abs() < 1e-6);
}

#[test]
fn test_functional_cross_entropy() {
    let logits = Tensor::from_vec(vec![2.0, 1.0, 0.1], &[1, 3]);
    let targets = Tensor::from_vec(vec![1.0, 0.0, 0.0], &[1, 3]);
    
    let loss = functional::cross_entropy_loss(&logits, &targets);
    assert_eq!(loss.shape(), &[1]);
}

#[test]
fn test_module_training_mode() {
    let mut linear = Linear::new(5, 3, true);
    
    // Default should be training mode
    assert!(linear.is_training());
    
    // Test eval mode
    linear.eval();
    assert!(!linear.is_training());
    
    // Test train mode
    linear.train();
    assert!(linear.is_training());
}

#[test]
fn test_module_display() {
    let linear = Linear::new(10, 5, true);
    let display_str = format!("{}", linear);
    assert!(display_str.contains("Linear"));
    assert!(display_str.contains("in_features=10"));
    assert!(display_str.contains("out_features=5"));
    assert!(display_str.contains("bias=true"));
    
    let sigmoid = Sigmoid::new();
    let display_str = format!("{}", sigmoid);
    assert!(display_str.contains("Sigmoid"));
    
    let relu = ReLU::new();
    let display_str = format!("{}", relu);
    assert!(display_str.contains("ReLU"));
}