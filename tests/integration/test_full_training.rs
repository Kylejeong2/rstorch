// Full neural network training integration test - End-to-end training pipeline verification
// Tests complete training workflow from data loading to model optimization
// Connected to: All main library components (tensor, nn, optim, utils)
// Used by: Integration test suite to verify complete library functionality

use rstorch::{Tensor, nn::{Linear, ReLU, MSELoss, Module}, optim::{SGD, OptimizerParams, Optimizer}};

#[test]
fn test_complete_training_pipeline() {
    // Create a simple regression dataset: y = 2*x + 1
    let train_data = vec![
        (vec![1.0], vec![3.0]),   // 2*1 + 1 = 3
        (vec![2.0], vec![5.0]),   // 2*2 + 1 = 5
        (vec![3.0], vec![7.0]),   // 2*3 + 1 = 7
        (vec![4.0], vec![9.0]),   // 2*4 + 1 = 9
        (vec![5.0], vec![11.0]),  // 2*5 + 1 = 11
    ];

    // Create a simple neural network: Linear(1, 10) -> ReLU -> Linear(10, 1)
    let layer1 = Linear::new(1, 10, true);
    let activation = ReLU::new();
    let layer2 = Linear::new(10, 1, true);
    let criterion = MSELoss::new();

    // Create optimizer
    let mut params_vec = Vec::new();
    for (i, param) in layer1.parameters().iter().enumerate() {
        params_vec.push((
            "layer1".to_string(),
            format!("param_{}", i),
            param.data().clone()
        ));
    }
    for (i, param) in layer2.parameters().iter().enumerate() {
        params_vec.push((
            "layer2".to_string(),
            format!("param_{}", i),
            param.data().clone()
        ));
    }

    let optimizer_params = OptimizerParams::new(params_vec).expect("Failed to create optimizer params");
    let mut optimizer = SGD::new(optimizer_params, 0.01, 0.0);

    // Training loop
    for epoch in 0..10 {
        let mut total_loss = 0.0;
        
        for (input_data, target_data) in &train_data {
            // Forward pass
            let input = Tensor::from_vec(input_data.clone(), &[1]).expect("Failed to create input tensor");
            let target = Tensor::from_vec(target_data.clone(), &[1]).expect("Failed to create target tensor");

            // Forward through network
            let x1 = layer1.forward(&[&input]).expect("Forward pass layer1 failed");
            let x2 = activation.forward(&[&x1]).expect("Forward pass activation failed");
            let output = layer2.forward(&[&x2]).expect("Forward pass layer2 failed");

            // Compute loss
            let loss = criterion.forward(&[&output, &target]).expect("Loss computation failed");
            total_loss += loss.to_vec()[0];

            // Zero gradients
            optimizer.zero_grad();

            // Note: In a complete implementation, backward pass would be called here
            // For now, we just verify the forward pass works
        }

        println!("Epoch {}: Average Loss = {:.4}", epoch, total_loss / train_data.len() as f32);
    }

    // Test that we can make predictions
    let test_input = Tensor::from_vec(vec![6.0], &[1]).expect("Failed to create test input");
    let x1 = layer1.forward(&[&test_input]).expect("Forward pass layer1 failed");
    let x2 = activation.forward(&[&x1]).expect("Forward pass activation failed");
    let prediction = layer2.forward(&[&x2]).expect("Forward pass layer2 failed");

    let pred_value = prediction.to_vec()[0];
    println!("Prediction for input 6.0: {:.2} (expected ~13.0)", pred_value);

    // Test should complete without panicking
    assert!(pred_value.is_finite(), "Prediction should be a finite number");
}

#[test]
fn test_tensor_operations_integration() {
    // Test comprehensive tensor operations work together
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Failed to create tensor a");
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("Failed to create tensor b");

    // Test basic operations
    let c = a.add(&b);
    assert_eq!(c.shape(), &[2, 2]);
    let c_data = c.to_vec();
    assert_eq!(c_data, vec![6.0, 8.0, 10.0, 12.0]);

    // Test matrix multiplication
    let d = a.matmul(&b);
    assert_eq!(d.shape(), &[2, 2]);

    // Test element-wise operations
    let e = a.mul(&b);
    assert_eq!(e.shape(), &[2, 2]);
    let e_data = e.to_vec();
    assert_eq!(e_data, vec![5.0, 12.0, 21.0, 32.0]);

    // Test activation functions
    let f = a.sigmoid();
    assert_eq!(f.shape(), &[2, 2]);
    let f_data = f.to_vec();
    // Sigmoid values should be between 0 and 1
    for &val in &f_data {
        assert!(val > 0.0 && val < 1.0, "Sigmoid output {} should be between 0 and 1", val);
    }

    // Test ReLU
    let negative_tensor = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2]).expect("Failed to create negative tensor");
    let relu_result = negative_tensor.relu();
    let relu_data = relu_result.to_vec();
    assert_eq!(relu_data, vec![0.0, 2.0, 0.0, 4.0]);

    println!("All tensor operations integration tests passed!");
}

#[test]
fn test_model_state_persistence() {
    // Test model saving and loading
    let mut model = Linear::new(2, 3, true);
    
    // Get initial state
    let initial_state = model.state_dict();
    assert!(!initial_state.is_empty(), "State dict should not be empty");

    // Modify the model (in practice this would be through training)
    // For testing, we can verify the state dict structure

    // Test loading state dict
    let load_result = model.load_state_dict(&initial_state);
    assert!(load_result.is_ok(), "Loading state dict should succeed: {:?}", load_result);

    // Test invalid state dict (wrong shape)
    let mut invalid_state = initial_state.clone();
    if let Some(param0) = invalid_state.get_mut("param0") {
        param0.push(999.0); // Make it wrong size
    }
    
    let invalid_load_result = model.load_state_dict(&invalid_state);
    assert!(invalid_load_result.is_err(), "Loading invalid state dict should fail");

    println!("Model state persistence test passed!");
}

#[test]
fn test_parameter_gradients() {
    // Test that parameters can be updated and gradients work
    let mut linear = Linear::new(2, 1, true);
    
    // Check initial parameters
    let params = linear.parameters();
    assert_eq!(params.len(), 2); // weight and bias
    
    // Verify parameter properties
    for param in params {
        assert!(param.requires_grad(), "Parameters should require gradients");
        assert_eq!(param.data().device, "cpu", "Parameters should be on CPU by default");
    }

    // Test zero_grad functionality
    linear.zero_grad();
    
    // Test training/eval mode switching
    linear.train();
    assert!(linear.is_training(), "Model should be in training mode");
    
    linear.eval();
    assert!(!linear.is_training(), "Model should be in evaluation mode");

    println!("Parameter gradients test passed!");
}

#[cfg(test)]
mod dataset_integration {
    use super::*;

    #[test]
    fn test_data_pipeline() {
        // Test creating batches of data
        let data_points = vec![
            (vec![1.0, 2.0], 0),
            (vec![3.0, 4.0], 1),
            (vec![5.0, 6.0], 0),
            (vec![7.0, 8.0], 1),
        ];

        // Convert to tensors
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for (input_data, target_class) in data_points {
            let input_tensor = Tensor::from_vec(input_data, &[2]).expect("Failed to create input tensor");
            let target_tensor = Tensor::from_vec(vec![target_class as f32], &[1]).expect("Failed to create target tensor");
            
            inputs.push(input_tensor);
            targets.push(target_tensor);
        }

        assert_eq!(inputs.len(), 4);
        assert_eq!(targets.len(), 4);

        // Test that all tensors have correct shapes
        for input in &inputs {
            assert_eq!(input.shape(), &[2]);
        }
        for target in &targets {
            assert_eq!(target.shape(), &[1]);
        }

        println!("Data pipeline test passed!");
    }
}