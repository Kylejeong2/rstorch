use std::os::raw::{c_char, c_float, c_int};
use std::ffi::CString;
use std::ptr;

#[repr(C)]
pub struct CTensor {
    pub data: *mut c_float,
    pub strides: *mut c_int,
    pub shape: *mut c_int,
    pub ndim: c_int,
    pub size: c_int,
    pub device: *const c_char,
}

// External C function declarations
extern "C" {
    fn create_tensor(data: *mut c_float, shape: *mut c_int, ndim: c_int, device: *const c_char) -> *mut CTensor;
    fn delete_tensor(tensor: *mut CTensor);
    fn delete_data(tensor: *mut CTensor);
    fn delete_shape(tensor: *mut CTensor);
    fn delete_strides(tensor: *mut CTensor);
    fn delete_device(tensor: *mut CTensor);
    
    fn ones_like_tensor(tensor: *mut CTensor) -> *mut CTensor;
    fn zeros_like_tensor(tensor: *mut CTensor) -> *mut CTensor;
    fn reshape_tensor(tensor: *mut CTensor, new_shape: *mut c_int, new_ndim: c_int) -> *mut CTensor;
    
    fn add_tensor(a: *mut CTensor, b: *mut CTensor) -> *mut CTensor;
    fn sub_tensor(a: *mut CTensor, b: *mut CTensor) -> *mut CTensor;
    fn elementwise_mul_tensor(a: *mut CTensor, b: *mut CTensor) -> *mut CTensor;
    fn scalar_mul_tensor(tensor: *mut CTensor, scalar: c_float) -> *mut CTensor;
    fn matmul_tensor(a: *mut CTensor, b: *mut CTensor) -> *mut CTensor;
    fn sum_tensor(tensor: *mut CTensor, axis: c_int, keepdim: bool) -> *mut CTensor;
    
    fn add_broadcasted_tensor(a: *mut CTensor, b: *mut CTensor) -> *mut CTensor;
    fn sub_broadcasted_tensor(a: *mut CTensor, b: *mut CTensor) -> *mut CTensor;
    
    fn get_item(tensor: *mut CTensor, indices: *mut c_int) -> c_float;
    fn to_device(tensor: *mut CTensor, device: *const c_char);
}

pub trait GradFn {
    fn backward(&self, grad: &Tensor) -> Vec<Tensor>;
}

pub struct Tensor {
    pub tensor: *mut CTensor,
    pub shape: Vec<usize>,
    pub ndim: usize,
    pub device: String,
    pub numel: usize,
    pub requires_grad: bool,
    pub grad: Option<Box<Tensor>>,
    pub grad_fn: Option<Box<dyn GradFn>>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("ndim", &self.ndim)
            .field("device", &self.device)
            .field("numel", &self.numel)
            .field("requires_grad", &self.requires_grad)
            .finish()
    }
}

impl Tensor {
    pub fn new(data: Option<Vec<f32>>, device: &str, requires_grad: bool) -> Self {
        if let Some(mut data_vec) = data {
            let shape = if data_vec.len() == 1 { vec![1] } else { vec![data_vec.len()] };
            
            let shape_ctype: Vec<c_int> = shape.iter().map(|&x| x as c_int).collect();
            let ndim_ctype = shape.len() as c_int;
            let device_cstring = CString::new(device).expect("Invalid device string");
            
            let numel = shape.iter().product();
            
            let tensor_ptr = unsafe {
                create_tensor(
                    data_vec.as_mut_ptr(),
                    shape_ctype.as_ptr() as *mut c_int,
                    ndim_ctype,
                    device_cstring.as_ptr(),
                )
            };
            
            Self {
                tensor: tensor_ptr,
                shape,
                ndim: ndim_ctype as usize,
                device: device.to_string(),
                numel,
                requires_grad,
                grad: None,
                grad_fn: None,
            }
        } else {
            Self {
                tensor: ptr::null_mut(),
                shape: vec![],
                ndim: 0,
                device: device.to_string(),
                numel: 0,
                requires_grad,
                grad: None,
                grad_fn: None,
            }
        }
    }
    
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        Self::from_vec_device(data, shape, "cpu")
    }
    
    pub fn from_vec_device(mut data: Vec<f32>, shape: &[usize], device: &str) -> Self {
        let shape_ctype: Vec<c_int> = shape.iter().map(|&x| x as c_int).collect();
        let ndim_ctype = shape.len() as c_int;
        let device_cstring = CString::new(device).expect("Invalid device string");
        
        let numel = shape.iter().product();
        
        let tensor_ptr = unsafe {
            create_tensor(
                data.as_mut_ptr(),
                shape_ctype.as_ptr() as *mut c_int,
                ndim_ctype,
                device_cstring.as_ptr(),
            )
        };
        
        Self {
            tensor: tensor_ptr,
            shape: shape.to_vec(),
            ndim: shape.len(),
            device: device.to_string(),
            numel,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        }
    }
    
    pub fn zeros(shape: &[usize]) -> Self {
        let data = vec![0.0; shape.iter().product()];
        Self::from_vec(data, shape)
    }
    
    pub fn ones_like(&self) -> Self {
        let result_tensor_ptr = unsafe { ones_like_tensor(self.tensor) };
        
        Self {
            tensor: result_tensor_ptr,
            shape: self.shape.clone(),
            ndim: self.ndim,
            device: self.device.clone(),
            numel: self.numel,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        }
    }
    
    pub fn zeros_like(&self) -> Self {
        let result_tensor_ptr = unsafe { zeros_like_tensor(self.tensor) };
        
        Self {
            tensor: result_tensor_ptr,
            shape: self.shape.clone(),
            ndim: self.ndim,
            device: self.device.clone(),
            numel: self.numel,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        }
    }
    
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Self {
        let result_tensor_ptr = unsafe { add_tensor(self.tensor, other.tensor) };
        
        Self {
            tensor: result_tensor_ptr,
            shape: self.shape.clone(),
            ndim: self.ndim,
            device: self.device.clone(),
            numel: self.numel,
            requires_grad: self.requires_grad || other.requires_grad,
            grad: None,
            grad_fn: None,
        }
    }
    
    /// Matrix multiplication
    pub fn matmul(&self, other: &Self) -> Self {
        if self.ndim != 2 || other.ndim != 2 {
            panic!("Matrix multiplication requires 2D tensors");
        }
        if self.shape[1] != other.shape[0] {
            panic!("Incompatible shapes for matrix multiplication");
        }
        
        let result_tensor_ptr = unsafe { matmul_tensor(self.tensor, other.tensor) };
        let result_shape = vec![self.shape[0], other.shape[1]];
        let result_numel = result_shape.iter().product();
        
        Self {
            tensor: result_tensor_ptr,
            shape: result_shape,
            ndim: 2,
            device: self.device.clone(),
            numel: result_numel,
            requires_grad: self.requires_grad || other.requires_grad,
            grad: None,
            grad_fn: None,
        }
    }
    
    /// Sum of all elements or along an axis
    pub fn sum(&self) -> Self {
        let result_tensor_ptr = unsafe { sum_tensor(self.tensor, -1, false) };
        
        Self {
            tensor: result_tensor_ptr,
            shape: vec![1],
            ndim: 1,
            device: self.device.clone(),
            numel: 1,
            requires_grad: self.requires_grad,
            grad: None,
            grad_fn: None,
        }
    }
    
    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> Self {
        let result_tensor_ptr = unsafe { elementwise_mul_tensor(self.tensor, other.tensor) };
        
        Self {
            tensor: result_tensor_ptr,
            shape: self.shape.clone(),
            ndim: self.ndim,
            device: self.device.clone(),
            numel: self.numel,
            requires_grad: self.requires_grad || other.requires_grad,
            grad: None,
            grad_fn: None,
        }
    }
    
    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: f32) -> Self {
        let result_tensor_ptr = unsafe { scalar_mul_tensor(self.tensor, scalar) };
        
        Self {
            tensor: result_tensor_ptr,
            shape: self.shape.clone(),
            ndim: self.ndim,
            device: self.device.clone(),
            numel: self.numel,
            requires_grad: self.requires_grad,
            grad: None,
            grad_fn: None,
        }
    }
}

// Implement standard operators
use std::ops::{Add, Mul};

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: f32) -> Self::Output {
        self.scalar_mul(rhs)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

// Memory management - cleanup C resources when Tensor is dropped
impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.tensor.is_null() {
            unsafe {
                delete_strides(self.tensor);
                delete_device(self.tensor);
                delete_tensor(self.tensor);
            }
        }
    }
} 