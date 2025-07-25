// Tensor C interface header - Core tensor structure and C API declarations
// Provides CTensor struct definition and C interface functions for tensor operations
// Connected to: src/csrc/tensor.cpp, src/csrc/tensor_impl.cpp, src/tensor.rs
// Used by: src/csrc/cpu.h, src/csrc/cuda.h, src/csrc/distributed.h, all C++ implementations

#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    const char* device;
} CTensor;

extern "C" {
    CTensor* create_tensor(float* data, int* shape, int ndim, const char* device);
    void delete_tensor(CTensor* tensor);
    void delete_data(CTensor* tensor);
    void delete_shape(CTensor* tensor);
    void delete_strides(CTensor* tensor);
    void delete_device(CTensor* tensor);
    
    float get_item(CTensor* tensor, int* indices);
    void to_device(CTensor* tensor, const char* device);
    float* get_data(CTensor* tensor);
    
    CTensor* add_tensor(CTensor* tensor1, CTensor* tensor2);
    CTensor* sub_tensor(CTensor* tensor1, CTensor* tensor2);
    CTensor* elementwise_mul_tensor(CTensor* tensor1, CTensor* tensor2);
    CTensor* scalar_mul_tensor(CTensor* tensor, float scalar);
    CTensor* matmul_tensor(CTensor* tensor1, CTensor* tensor2);
    CTensor* sum_tensor(CTensor* tensor, int axis, bool keepdim);
    
    CTensor* add_broadcasted_tensor(CTensor* tensor1, CTensor* tensor2);
    CTensor* sub_broadcasted_tensor(CTensor* tensor1, CTensor* tensor2);
    
    CTensor* sigmoid_tensor(CTensor* tensor);
    CTensor* softmax_tensor(CTensor* tensor, int axis);
    CTensor* relu_tensor(CTensor* tensor);
    CTensor* log_tensor(CTensor* tensor);
    
    CTensor* ones_like_tensor(CTensor* tensor);
    CTensor* zeros_like_tensor(CTensor* tensor);
    CTensor* reshape_tensor(CTensor* tensor, int* new_shape, int new_ndim);
}

#endif /* TENSOR_H */