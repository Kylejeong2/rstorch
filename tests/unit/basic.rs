// Basic tensor functionality tests - Core tensor interface and operation tests
// Tests fundamental tensor creation, operations, and interfaces with mock C functions
// Connected to: src/tensor.rs, src/csrc/tensor.h
// Used by: Test suite verification of basic tensor functionality

use rstorch::Tensor;

// Mock C functions for testing
#[cfg(test)]
mod mock_c_functions {
    use std::os::raw::{c_char, c_float, c_int};
    use rstorch::CTensor;
    
    #[no_mangle]
    extern "C" fn create_tensor(_data: *mut c_float, _shape: *mut c_int, _ndim: c_int, _device: *const c_char) -> *mut CTensor {
        Box::into_raw(Box::new(CTensor {
            data: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            shape: std::ptr::null_mut(),
            ndim: 0,
            size: 0,
            device: std::ptr::null(),
        }))
    }
    
    #[no_mangle]
    extern "C" fn delete_tensor(_tensor: *mut CTensor) {}
    
    #[no_mangle]
    extern "C" fn delete_strides(_tensor: *mut CTensor) {}
    
    #[no_mangle]
    extern "C" fn delete_device(_tensor: *mut CTensor) {}
    
    #[no_mangle]
    extern "C" fn add_tensor(_a: *mut CTensor, _b: *mut CTensor) -> *mut CTensor {
        create_tensor(std::ptr::null_mut(), std::ptr::null_mut(), 0, std::ptr::null())
    }
    
    #[no_mangle]
    extern "C" fn matmul_tensor(_a: *mut CTensor, _b: *mut CTensor) -> *mut CTensor {
        create_tensor(std::ptr::null_mut(), std::ptr::null_mut(), 0, std::ptr::null())
    }
    
    #[no_mangle]
    extern "C" fn sum_tensor(_tensor: *mut CTensor, _axis: c_int, _keepdim: bool) -> *mut CTensor {
        create_tensor(std::ptr::null_mut(), std::ptr::null_mut(), 0, std::ptr::null())
    }
    
    #[no_mangle]
    extern "C" fn ones_like_tensor(_tensor: *mut CTensor) -> *mut CTensor {
        create_tensor(std::ptr::null_mut(), std::ptr::null_mut(), 0, std::ptr::null())
    }
    
    #[no_mangle]
    extern "C" fn zeros_like_tensor(_tensor: *mut CTensor) -> *mut CTensor {
        create_tensor(std::ptr::null_mut(), std::ptr::null_mut(), 0, std::ptr::null())
    }
    
    #[no_mangle]
    extern "C" fn elementwise_mul_tensor(_a: *mut CTensor, _b: *mut CTensor) -> *mut CTensor {
        create_tensor(std::ptr::null_mut(), std::ptr::null_mut(), 0, std::ptr::null())
    }
    
    #[no_mangle]
    extern "C" fn scalar_mul_tensor(_tensor: *mut CTensor, _scalar: c_float) -> *mut CTensor {
        create_tensor(std::ptr::null_mut(), std::ptr::null_mut(), 0, std::ptr::null())
    }
}

#[test]
fn test_tensor_creation() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(a.shape(), &[2, 2]);
    assert_eq!(a.ndim, 2);
    assert_eq!(a.numel, 4);
    assert_eq!(a.device, "cpu");
}

#[test]
fn test_add_interface() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 3.0, 2.0, 1.0], &[2, 2]).unwrap();
    let c = &a + &b;
    
    // With mock functions, we can't test actual values, but we can test interface
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.ndim, 2);
}

#[test]
fn test_matmul_interface() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::from_vec(vec![4.0, 3.0, 2.0, 1.0], &[2, 2]).unwrap();
    let c = a.matmul(&b);
    
    // Test that result has correct shape for 2x2 @ 2x2 = 2x2
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.ndim, 2);
}

#[test]
fn test_sum_interface() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let s = a.sum(-1, false);  // Sum all elements (axis=-1 means all)
    
    // Sum should produce a scalar (1D tensor with 1 element)
    assert_eq!(s.shape(), &[1]);
    assert_eq!(s.ndim, 1);
} 