// Unit tests for optimizer components - Tests optimizers and parameter management in isolation
// Verifies optimizer creation, parameter handling, and update mechanisms work correctly
// Connected to: src/optim/
// Used by: Unit test suite to verify optimizer functionality

use rstorch::{
    Tensor,
    optim::{SGD, OptimizerParams, Optimizer},
    nn::{Linear, Parameter}
};
use std::collections::HashMap;

#[test]
fn test_optimizer_params_creation() {
    // Test creating OptimizerParams from iterator
    let params_vec = vec![
        ("module1".to_string(), "weight".to_string(), Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Tensor creation failed")),
        ("module1".to_string(), "bias".to_string(), Tensor::from_vec(vec![0.1], &[1]).expect("Tensor creation failed")),
        ("module2".to_string(), "weight".to_string(), Tensor::from_vec(vec![3.0, 4.0, 5.0, 6.0], &[2, 2]).expect("Tensor creation failed")),
    ];

    let optimizer_params = OptimizerParams::new(params_vec.clone());
    assert!(optimizer_params.is_ok(), "OptimizerParams creation should succeed");
    
    let optimizer_params = optimizer_params.unwrap();
    assert_eq!(optimizer_params.parameters.len(), 3, "Should have 3 parameters");
    
    // Check that parameters were stored correctly
    assert_eq!(optimizer_params.parameters[0].0, "module1", "First parameter module name should be correct");
    assert_eq!(optimizer_params.parameters[0].1, "weight", "First parameter name should be correct");
    assert_eq!(optimizer_params.parameters[0].2.shape(), &[2], "First parameter shape should be correct");
}

#[test]
fn test_optimizer_params_from_dict() {
    let mut param_dict = HashMap::new();
    param_dict.insert("weight1".to_string(), Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Tensor creation failed"));
    param_dict.insert("bias1".to_string(), Tensor::from_vec(vec![0.5], &[1]).expect("Tensor creation failed"));

    let optimizer_params = OptimizerParams::from_dict(param_dict);
    assert_eq!(optimizer_params.parameters.len(), 2, "Should have 2 parameters from dict");
    
    // Check that parameters were stored (order might vary due to HashMap)
    let param_names: Vec<&String> = optimizer_params.parameters.iter().map(|(_, name, _)| name).collect();
    assert!(param_names.contains(&&"weight1".to_string()), "Should contain weight1");
    assert!(param_names.contains(&&"bias1".to_string()), "Should contain bias1");
}

#[test]
fn test_optimizer_params_zero_grad() {
    let params_vec = vec![
        ("module".to_string(), "param1".to_string(), Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Tensor creation failed")),
        ("module".to_string(), "param2".to_string(), Tensor::from_vec(vec![3.0], &[1]).expect("Tensor creation failed")),
    ];

    let mut optimizer_params = OptimizerParams::new(params_vec).expect("OptimizerParams creation failed");
    
    // zero_grad should not panic
    optimizer_params.zero_grad();
    
    // Note: In a complete implementation, we would verify that gradients are actually zeroed
    // For now, we just test that the method exists and doesn't panic
}

#[test]
fn test_sgd_creation() {
    let params_vec = vec![
        ("layer".to_string(), "weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Tensor creation failed")),
        ("layer".to_string(), "bias".to_string(), Tensor::from_vec(vec![0.1, 0.2], &[2]).expect("Tensor creation failed")),
    ];

    let optimizer_params = OptimizerParams::new(params_vec).expect("OptimizerParams creation failed");
    
    // Test SGD creation with different parameters
    let sgd1 = SGD::new(optimizer_params, 0.01, 0.0);
    assert_eq!(sgd1.lr, 0.01, "Learning rate should be set correctly");
    assert_eq!(sgd1.momentum, 0.0, "Momentum should be set correctly");
    assert_eq!(sgd1.velocity_cache.len(), 2, "Velocity cache should have same length as parameters");
    
    // Test with momentum
    let params_vec2 = vec![
        ("layer".to_string(), "weight".to_string(), Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Tensor creation failed")),
    ];
    let optimizer_params2 = OptimizerParams::new(params_vec2).expect("OptimizerParams creation failed");
    
    let sgd2 = SGD::new(optimizer_params2, 0.1, 0.9);
    assert_eq!(sgd2.lr, 0.1, "Learning rate should be set correctly");
    assert_eq!(sgd2.momentum, 0.9, "Momentum should be set correctly");
}

#[test]
fn test_sgd_parameter_shapes() {
    // Test that velocity cache has correct shapes
    let params_vec = vec![
        ("layer".to_string(), "weight".to_string(), Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("Tensor creation failed")),
        ("layer".to_string(), "bias".to_string(), Tensor::from_vec(vec![0.1, 0.2], &[2]).expect("Tensor creation failed")),
    ];

    let optimizer_params = OptimizerParams::new(params_vec).expect("OptimizerParams creation failed");
    let sgd = SGD::new(optimizer_params, 0.01, 0.9);
    
    // Check velocity cache shapes match parameter shapes
    assert_eq!(sgd.velocity_cache[0].shape(), &[2, 2], "First velocity should match first parameter shape");
    assert_eq!(sgd.velocity_cache[1].shape(), &[2], "Second velocity should match second parameter shape");
    
    // Check that velocity cache is initialized to zeros
    let velocity_data = sgd.velocity_cache[0].to_vec();
    for &val in &velocity_data {
        assert_eq!(val, 0.0, "Velocity cache should be initialized to zeros");
    }
}

#[test]
fn test_sgd_optimizer_trait() {
    let params_vec = vec![
        ("layer".to_string(), "param".to_string(), Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Tensor creation failed")),
    ];

    let optimizer_params = OptimizerParams::new(params_vec).expect("OptimizerParams creation failed");
    let mut sgd = SGD::new(optimizer_params, 0.01, 0.0);
    
    // Test that SGD implements Optimizer trait
    sgd.zero_grad(); // Should not panic
    sgd.step();      // Should not panic
    
    // Note: In a complete implementation with actual gradients, we would test that
    // parameters are actually updated. For now, we just test the interface exists.
}

#[test]
fn test_sgd_with_different_learning_rates() {
    let params_vec = vec![
        ("layer".to_string(), "param".to_string(), Tensor::from_vec(vec![1.0], &[1]).expect("Tensor creation failed")),
    ];

    // Test very small learning rate
    let optimizer_params1 = OptimizerParams::new(params_vec.clone()).expect("OptimizerParams creation failed");
    let sgd_small = SGD::new(optimizer_params1, 1e-6, 0.0);
    assert_eq!(sgd_small.lr, 1e-6, "Small learning rate should be set correctly");
    
    // Test large learning rate
    let optimizer_params2 = OptimizerParams::new(params_vec.clone()).expect("OptimizerParams creation failed");
    let sgd_large = SGD::new(optimizer_params2, 10.0, 0.0);
    assert_eq!(sgd_large.lr, 10.0, "Large learning rate should be set correctly");
    
    // Test zero learning rate
    let optimizer_params3 = OptimizerParams::new(params_vec).expect("OptimizerParams creation failed");
    let sgd_zero = SGD::new(optimizer_params3, 0.0, 0.0);
    assert_eq!(sgd_zero.lr, 0.0, "Zero learning rate should be set correctly");
}

#[test]
fn test_sgd_with_different_momentum() {
    let params_vec = vec![
        ("layer".to_string(), "param".to_string(), Tensor::from_vec(vec![1.0], &[1]).expect("Tensor creation failed")),
    ];

    // Test no momentum
    let optimizer_params1 = OptimizerParams::new(params_vec.clone()).expect("OptimizerParams creation failed");
    let sgd_no_momentum = SGD::new(optimizer_params1, 0.01, 0.0);
    assert_eq!(sgd_no_momentum.momentum, 0.0, "No momentum should be 0.0");
    
    // Test high momentum
    let optimizer_params2 = OptimizerParams::new(params_vec.clone()).expect("OptimizerParams creation failed");
    let sgd_high_momentum = SGD::new(optimizer_params2, 0.01, 0.99);
    assert_eq!(sgd_high_momentum.momentum, 0.99, "High momentum should be set correctly");
    
    // Test typical momentum
    let optimizer_params3 = OptimizerParams::new(params_vec).expect("OptimizerParams creation failed");
    let sgd_typical = SGD::new(optimizer_params3, 0.01, 0.9);
    assert_eq!(sgd_typical.momentum, 0.9, "Typical momentum should be set correctly");
}

#[test]
fn test_optimizer_with_actual_model() {
    // Test creating optimizer from actual model parameters
    let mut layer = Linear::new(2, 1, true);
    
    // Extract parameters from the model
    let mut params_vec = Vec::new();
    for (i, param) in layer.parameters().iter().enumerate() {
        params_vec.push((
            "linear".to_string(),
            format!("param_{}", i),
            param.data().clone()
        ));
    }
    
    let optimizer_params = OptimizerParams::new(params_vec).expect("OptimizerParams creation failed");
    let mut optimizer = SGD::new(optimizer_params, 0.01, 0.9);
    
    // Test that we can call optimizer methods
    optimizer.zero_grad();
    optimizer.step();
    
    // Verify optimizer has correct number of parameters
    assert_eq!(optimizer.params.parameters.len(), 2, "Should have weight and bias parameters");
    assert_eq!(optimizer.velocity_cache.len(), 2, "Should have velocity for each parameter");
}

#[test]
fn test_optimizer_params_parameter_access() {
    let tensor1 = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Tensor creation failed");
    let tensor2 = Tensor::from_vec(vec![3.0], &[1]).expect("Tensor creation failed");
    
    let params_vec = vec![
        ("module1".to_string(), "weight".to_string(), tensor1.clone()),
        ("module1".to_string(), "bias".to_string(), tensor2.clone()),
    ];

    let optimizer_params = OptimizerParams::new(params_vec).expect("OptimizerParams creation failed");
    
    // Test parameter access
    assert_eq!(optimizer_params.parameters.len(), 2, "Should have 2 parameters");
    
    let (module_name, param_name, tensor) = &optimizer_params.parameters[0];
    assert_eq!(module_name, "module1", "Module name should be correct");
    assert_eq!(param_name, "weight", "Parameter name should be correct");
    assert_eq!(tensor.shape(), &[2], "Tensor shape should be correct");
    assert_eq!(tensor.to_vec(), vec![1.0, 2.0], "Tensor data should be correct");
}

#[test]
fn test_empty_optimizer_params() {
    // Test creating optimizer with no parameters
    let empty_params: Vec<(String, String, Tensor)> = vec![];
    let optimizer_params = OptimizerParams::new(empty_params).expect("Empty OptimizerParams creation should succeed");
    
    assert_eq!(optimizer_params.parameters.len(), 0, "Empty params should have 0 parameters");
    
    // Test creating SGD with empty parameters
    let sgd = SGD::new(optimizer_params, 0.01, 0.0);
    assert_eq!(sgd.params.parameters.len(), 0, "SGD should handle empty parameters");
    assert_eq!(sgd.velocity_cache.len(), 0, "Velocity cache should be empty");
}

#[test]
fn test_optimizer_parameter_data_integrity() {
    // Test that optimizer doesn't modify original parameter data during creation
    let original_data = vec![1.0, 2.0, 3.0];
    let tensor = Tensor::from_vec(original_data.clone(), &[3]).expect("Tensor creation failed");
    
    let params_vec = vec![
        ("module".to_string(), "param".to_string(), tensor.clone()),
    ];

    let optimizer_params = OptimizerParams::new(params_vec).expect("OptimizerParams creation failed");
    let _sgd = SGD::new(optimizer_params, 0.01, 0.0);
    
    // Original tensor should remain unchanged
    assert_eq!(tensor.to_vec(), original_data, "Original tensor data should be unchanged");
}

#[test]
fn test_multiple_optimizers() {
    // Test creating multiple optimizers with different configurations
    let params_vec1 = vec![
        ("layer1".to_string(), "weight".to_string(), Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("Tensor creation failed")),
    ];
    let params_vec2 = vec![
        ("layer2".to_string(), "weight".to_string(), Tensor::from_vec(vec![3.0, 4.0], &[2]).expect("Tensor creation failed")),
    ];

    let optimizer_params1 = OptimizerParams::new(params_vec1).expect("OptimizerParams1 creation failed");
    let optimizer_params2 = OptimizerParams::new(params_vec2).expect("OptimizerParams2 creation failed");
    
    let mut sgd1 = SGD::new(optimizer_params1, 0.01, 0.9);
    let mut sgd2 = SGD::new(optimizer_params2, 0.1, 0.0);
    
    // Test that optimizers are independent
    assert_eq!(sgd1.lr, 0.01, "First optimizer should have correct learning rate");
    assert_eq!(sgd2.lr, 0.1, "Second optimizer should have correct learning rate");
    assert_eq!(sgd1.momentum, 0.9, "First optimizer should have correct momentum");
    assert_eq!(sgd2.momentum, 0.0, "Second optimizer should have correct momentum");
    
    // Test that they can be used independently
    sgd1.zero_grad();
    sgd2.zero_grad();
    sgd1.step();
    sgd2.step();
}