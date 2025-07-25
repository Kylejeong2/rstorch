// Unit tests for neural network components - Tests individual NN components in isolation
// Verifies layers, activations, losses, and parameters work correctly independently
// Connected to: src/nn/
// Used by: Unit test suite to verify neural network component functionality

use rstorch::{
    Tensor,
    nn::{Linear, ReLU, Sigmoid, Softmax, MSELoss, CrossEntropyLoss, Module, Parameter, ParameterTensor}
};

#[test]
fn test_parameter_tensor_creation() {
    let param = ParameterTensor::new(&[2, 3]);
    
    assert_eq!(param.shape(), &[2, 3], "Parameter should have correct shape");
    assert!(param.requires_grad(), "Parameter should require gradients by default");
    assert_eq!(param.data().device, "cpu", "Parameter should be on CPU by default");
    
    // Test that data is random (not all zeros)
    let data = param.data().to_vec();
    let all_zeros = data.iter().all(|&x| x == 0.0);
    assert!(!all_zeros, "Parameter data should be randomly initialized, not all zeros");
}

#[test]
fn test_parameter_tensor_from_tensor() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Tensor creation failed");
    let param = ParameterTensor::from_tensor(tensor);
    
    assert_eq!(param.shape(), &[3], "Parameter should inherit tensor shape");
    let param_data = param.data().to_vec();
    assert_eq!(param_data, vec![1.0, 2.0, 3.0], "Parameter should have same data as source tensor");
}

#[test]
fn test_parameter_tensor_methods() {
    let mut param = ParameterTensor::new(&[2, 2]);
    
    // Test requires_grad
    assert!(param.requires_grad(), "Parameter should require gradients initially");
    param.set_requires_grad(false);
    assert!(!param.requires_grad(), "Parameter should not require gradients after setting false");
    
    // Test zero_grad
    param.zero_grad(); // Should not panic
    
    // Test to_device
    param.to_device("cpu"); // Should not panic
    assert_eq!(param.data().device, "cpu", "Parameter device should be updated");
    
    // Test data access
    let data = param.data();
    assert_eq!(data.shape(), &[2, 2], "Data should have correct shape");
    
    let data_mut = param.data_mut();
    assert_eq!(data_mut.shape(), &[2, 2], "Mutable data should have correct shape");
}

#[test]
fn test_parameter_set_data() {
    let mut param = ParameterTensor::new(&[2]);
    
    // Test successful data update
    let new_data = vec![5.0, 10.0];
    let result = param.set_data(new_data.clone());
    assert!(result.is_ok(), "Setting valid data should succeed");
    
    let updated_data = param.data().to_vec();
    assert_eq!(updated_data, new_data, "Parameter data should be updated");
    
    // Test invalid data length
    let invalid_data = vec![1.0, 2.0, 3.0]; // Wrong size
    let result = param.set_data(invalid_data);
    assert!(result.is_err(), "Setting invalid data length should fail");
}

#[test]
fn test_linear_layer_creation() {
    // Test with bias
    let layer_with_bias = Linear::new(3, 2, true);
    let params = layer_with_bias.parameters();
    assert_eq!(params.len(), 2, "Layer with bias should have 2 parameters");
    
    // Check parameter shapes
    assert_eq!(params[0].shape(), &[2, 3], "Weight should have shape [output, input]");
    assert_eq!(params[1].shape(), &[2], "Bias should have shape [output]");
    
    // Test without bias
    let layer_no_bias = Linear::new(3, 2, false);
    let params_no_bias = layer_no_bias.parameters();
    assert_eq!(params_no_bias.len(), 1, "Layer without bias should have 1 parameter");
    assert_eq!(params_no_bias[0].shape(), &[2, 3], "Weight should have shape [output, input]");
}

#[test]
fn test_linear_layer_forward() {
    let layer = Linear::new(2, 3, true);
    
    // Test 1D input
    let input_1d = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Input tensor creation failed");
    let output_1d = layer.forward(&[&input_1d]).expect("Forward pass should succeed");
    assert_eq!(output_1d.shape(), &[3], "1D output should have correct shape");
    
    // Test 2D batch input
    let input_2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Batch input creation failed");
    let output_2d = layer.forward(&[&input_2d]).expect("Batch forward pass should succeed");
    assert_eq!(output_2d.shape(), &[2, 3], "2D output should have correct shape");
    
    // Test error case: wrong number of inputs
    let result = layer.forward(&[]);
    assert!(result.is_err(), "Forward with no inputs should fail");
    
    let result = layer.forward(&[&input_1d, &input_1d]);
    assert!(result.is_err(), "Forward with too many inputs should fail");
}

#[test]
fn test_linear_layer_display() {
    let layer = Linear::new(4, 2, true);
    let display_str = format!("{}", layer);
    assert!(display_str.contains("Linear"), "Display should contain 'Linear'");
    assert!(display_str.contains("4"), "Display should contain input dimension");
    assert!(display_str.contains("2"), "Display should contain output dimension");
    assert!(display_str.contains("true"), "Display should indicate bias=true");
    
    let layer_no_bias = Linear::new(3, 1, false);
    let display_str_no_bias = format!("{}", layer_no_bias);
    assert!(display_str_no_bias.contains("false"), "Display should indicate bias=false");
}

#[test]
fn test_relu_activation() {
    let relu = ReLU::new();
    
    // Test with mixed positive/negative values
    let input = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).expect("Input creation failed");
    let output = relu.forward(&[&input]).expect("ReLU forward should succeed");
    
    let output_data = output.to_vec();
    assert_eq!(output_data, vec![0.0, 0.0, 0.0, 1.0, 2.0], "ReLU should zero negative values");
    
    // Test error case: wrong number of inputs
    let result = relu.forward(&[]);
    assert!(result.is_err(), "ReLU with no inputs should fail");
}

#[test]
fn test_sigmoid_activation() {
    let sigmoid = Sigmoid::new();
    
    let input = Tensor::from_vec(vec![-10.0, 0.0, 10.0], &[3]).expect("Input creation failed");
    let output = sigmoid.forward(&[&input]).expect("Sigmoid forward should succeed");
    
    let output_data = output.to_vec();
    
    // Check bounds
    for &val in &output_data {
        assert!(val > 0.0 && val < 1.0, "Sigmoid output {} should be between 0 and 1", val);
    }
    
    // Check specific values
    assert!(output_data[0] < 0.01, "Sigmoid(-10) should be close to 0");
    assert!((output_data[1] - 0.5).abs() < 1e-6, "Sigmoid(0) should be 0.5");
    assert!(output_data[2] > 0.99, "Sigmoid(10) should be close to 1");
}

#[test]
fn test_softmax_activation() {
    let softmax = Softmax::new(-1);
    
    let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Input creation failed");
    let output = softmax.forward(&[&input]).expect("Softmax forward should succeed");
    
    let output_data = output.to_vec();
    
    // Check that outputs sum to 1
    let sum: f32 = output_data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6, "Softmax outputs should sum to 1");
    
    // Check that all outputs are positive
    for &val in &output_data {
        assert!(val > 0.0, "Softmax output {} should be positive", val);
    }
    
    // Check ordering preservation
    assert!(output_data[2] > output_data[1], "Larger input should produce larger output");
    assert!(output_data[1] > output_data[0], "Larger input should produce larger output");
}

#[test]
fn test_mse_loss() {
    let mse_loss = MSELoss::new();
    
    // Test exact match (zero loss)
    let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Predictions creation failed");
    let targets = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("Targets creation failed");
    
    let loss = mse_loss.forward(&[&predictions, &targets]).expect("MSE loss should succeed");
    let loss_value = loss.to_vec()[0];
    assert!((loss_value - 0.0).abs() < 1e-6, "MSE loss for identical tensors should be 0");
    
    // Test with actual difference
    let targets_diff = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).expect("Different targets creation failed");
    let loss_diff = mse_loss.forward(&[&predictions, &targets_diff]).expect("MSE loss should succeed");
    let loss_diff_value = loss_diff.to_vec()[0];
    
    // Expected: ((1-2)^2 + (2-3)^2 + (3-4)^2) / 3 = (1 + 1 + 1) / 3 = 1.0
    assert!((loss_diff_value - 1.0).abs() < 1e-6, "MSE loss should be 1.0, got {}", loss_diff_value);
    
    // Test error case: wrong number of inputs
    let result = mse_loss.forward(&[&predictions]);
    assert!(result.is_err(), "MSE loss with one input should fail");
}

#[test]
fn test_cross_entropy_loss() {
    let ce_loss = CrossEntropyLoss::new();
    
    // Test with simple case
    let logits = Tensor::from_vec(vec![2.0, 1.0, 0.1], &[3]).expect("Logits creation failed");
    let targets = Tensor::from_vec(vec![1.0, 0.0, 0.0], &[3]).expect("Targets creation failed");
    
    let loss = ce_loss.forward(&[&logits, &targets]).expect("CrossEntropy loss should succeed");
    let loss_value = loss.to_vec()[0];
    
    assert!(loss_value >= 0.0, "CrossEntropy loss should be non-negative");
    assert!(loss_value.is_finite(), "CrossEntropy loss should be finite");
    
    // Test error case: wrong number of inputs
    let result = ce_loss.forward(&[&logits]);
    assert!(result.is_err(), "CrossEntropy loss with one input should fail");
}

#[test]
fn test_module_training_states() {
    let mut relu = ReLU::new();
    
    // Test initial state
    assert!(relu.is_training(), "Module should start in training mode");
    
    // Test state changes
    relu.eval();
    assert!(!relu.is_training(), "Module should be in eval mode");
    
    relu.train();
    assert!(relu.is_training(), "Module should be in training mode");
    
    // Test set_training directly
    relu.set_training(false);
    assert!(!relu.is_training(), "Module should respond to set_training(false)");
    
    relu.set_training(true);
    assert!(relu.is_training(), "Module should respond to set_training(true)");
}

#[test]
fn test_module_parameters() {
    let layer = Linear::new(2, 3, true);
    
    // Test parameters access
    let params = layer.parameters();
    assert_eq!(params.len(), 2, "Layer should have 2 parameters");
    
    for param in params {
        assert!(param.requires_grad(), "All parameters should require gradients");
    }
    
    // Test modules (should be empty for leaf modules)
    let modules = layer.modules();
    assert_eq!(modules.len(), 0, "Leaf module should have no sub-modules");
}

#[test]
fn test_module_zero_grad() {
    let mut layer = Linear::new(2, 1, true);
    
    // zero_grad should not panic
    layer.zero_grad();
    
    // Test that it works with different modules
    let mut relu = ReLU::new();
    relu.zero_grad(); // Should not panic even though ReLU has no parameters
}

#[test]
fn test_module_display() {
    let relu = ReLU::new();
    let display_str = format!("{}", relu);
    assert_eq!(display_str, "ReLU()", "ReLU display should be correct");
    
    let sigmoid = Sigmoid::new();
    let sigmoid_str = format!("{}", sigmoid);
    assert_eq!(sigmoid_str, "Sigmoid()", "Sigmoid display should be correct");
    
    let softmax = Softmax::new(1);
    let softmax_str = format!("{}", softmax);
    assert!(softmax_str.contains("Softmax"), "Softmax display should contain 'Softmax'");
    assert!(softmax_str.contains("1"), "Softmax display should contain dimension");
}

#[test]
fn test_loss_display() {
    let mse = MSELoss::new();
    let mse_str = format!("{}", mse);
    assert!(mse_str.contains("MSE"), "MSE display should contain 'MSE'");
    
    let ce = CrossEntropyLoss::new();
    let ce_str = format!("{}", ce);
    assert!(ce_str.contains("CrossEntropy"), "CrossEntropy display should contain 'CrossEntropy'");
}

#[test]
fn test_activation_trait() {
    // Test that activations implement the Activation trait
    let relu = ReLU::new();
    let sigmoid = Sigmoid::new();
    let softmax = Softmax::new(-1);
    
    // These should compile without error, proving they implement Activation
    use rstorch::nn::Activation;
    fn test_activation<T: Activation>(_: &T) {}
    
    test_activation(&relu);
    test_activation(&sigmoid);
    test_activation(&softmax);
}

#[test]
fn test_module_inner_repr() {
    let layer = Linear::new(4, 2, true);
    let inner_repr = layer.inner_repr();
    
    assert!(inner_repr.contains("in_features=4"), "Inner repr should contain input features");
    assert!(inner_repr.contains("out_features=2"), "Inner repr should contain output features");
    assert!(inner_repr.contains("bias=true"), "Inner repr should contain bias info");
}

#[test]
fn test_parameter_device_handling() {
    let mut param = ParameterTensor::new(&[2, 2]);
    
    // Test initial device
    assert_eq!(param.data().device, "cpu", "Parameter should start on CPU");
    
    // Test device change
    param.to_device("cpu"); // Should not panic
    assert_eq!(param.data().device, "cpu", "Device should remain CPU");
    
    // Note: Testing GPU device would require actual GPU support
}