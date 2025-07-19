#include "tensor.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

extern "C" {

CTensor* create_tensor(float* data, int* shape, int ndim, const char* device) {
    CTensor* tensor = new CTensor();
    
    // Calculate total size
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    tensor->size = size;
    tensor->ndim = ndim;
    
    // Allocate and copy data
    tensor->data = new float[size];
    if (data != nullptr) {
        std::memcpy(tensor->data, data, size * sizeof(float));
    } else {
        std::fill(tensor->data, tensor->data + size, 0.0f);
    }
    
    // Allocate and copy shape
    tensor->shape = new int[ndim];
    std::memcpy(tensor->shape, shape, ndim * sizeof(int));
    
    // Calculate strides
    tensor->strides = new int[ndim];
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }
    
    // Copy device string
    tensor->device = device;
    
    return tensor;
}

void delete_tensor(CTensor* tensor) {
    if (tensor != nullptr) {
        delete tensor;
    }
}

void delete_data(CTensor* tensor) {
    if (tensor != nullptr && tensor->data != nullptr) {
        delete[] tensor->data;
        tensor->data = nullptr;
    }
}

void delete_shape(CTensor* tensor) {
    if (tensor != nullptr && tensor->shape != nullptr) {
        delete[] tensor->shape;
        tensor->shape = nullptr;
    }
}

void delete_strides(CTensor* tensor) {
    if (tensor != nullptr && tensor->strides != nullptr) {
        delete[] tensor->strides;
        tensor->strides = nullptr;
    }
}

void delete_device(CTensor* /* tensor */) {
    // Device is const char*, no need to delete
}

CTensor* ones_like_tensor(CTensor* tensor) {
    CTensor* result = create_tensor(nullptr, tensor->shape, tensor->ndim, tensor->device);
    std::fill(result->data, result->data + result->size, 1.0f);
    return result;
}

CTensor* zeros_like_tensor(CTensor* tensor) {
    CTensor* result = create_tensor(nullptr, tensor->shape, tensor->ndim, tensor->device);
    std::fill(result->data, result->data + result->size, 0.0f);
    return result;
}

CTensor* reshape_tensor(CTensor* tensor, int* new_shape, int new_ndim) {
    // Calculate new size to verify it matches
    int new_size = 1;
    for (int i = 0; i < new_ndim; i++) {
        new_size *= new_shape[i];
    }
    
    if (new_size != tensor->size) {
        return nullptr; // Invalid reshape
    }
    
    CTensor* result = create_tensor(tensor->data, new_shape, new_ndim, tensor->device);
    return result;
}

float get_item(CTensor* tensor, int* indices) {
    int flat_idx = 0;
    for (int i = 0; i < tensor->ndim; i++) {
        flat_idx += indices[i] * tensor->strides[i];
    }
    return tensor->data[flat_idx];
}

void to_device(CTensor* tensor, const char* device) {
    tensor->device = device;
}

float* get_data(CTensor* tensor) {
    return tensor->data;
}

// Arithmetic operations
CTensor* add_tensor(CTensor* a, CTensor* b) {
    CTensor* result = create_tensor(nullptr, a->shape, a->ndim, a->device);
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    return result;
}

CTensor* sub_tensor(CTensor* a, CTensor* b) {
    CTensor* result = create_tensor(nullptr, a->shape, a->ndim, a->device);
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    return result;
}

CTensor* elementwise_mul_tensor(CTensor* a, CTensor* b) {
    CTensor* result = create_tensor(nullptr, a->shape, a->ndim, a->device);
    for (int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    return result;
}

CTensor* scalar_mul_tensor(CTensor* tensor, float scalar) {
    CTensor* result = create_tensor(nullptr, tensor->shape, tensor->ndim, tensor->device);
    for (int i = 0; i < tensor->size; i++) {
        result->data[i] = tensor->data[i] * scalar;
    }
    return result;
}

CTensor* matmul_tensor(CTensor* a, CTensor* b) {
    // Assuming 2D matrices for simplicity
    int m = a->shape[0];
    int n = a->shape[1];
    int p = b->shape[1];
    
    int result_shape[] = {m, p};
    CTensor* result = create_tensor(nullptr, result_shape, 2, a->device);
    
    // Simple matrix multiplication
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a->data[i * n + k] * b->data[k * p + j];
            }
            result->data[i * p + j] = sum;
        }
    }
    
    return result;
}

CTensor* sum_tensor(CTensor* tensor, int axis, bool /* keepdim */) {
    if (axis == -1) {
        // Sum all elements
        int result_shape[] = {1};
        CTensor* result = create_tensor(nullptr, result_shape, 1, tensor->device);
        float sum = 0.0f;
        for (int i = 0; i < tensor->size; i++) {
            sum += tensor->data[i];
        }
        result->data[0] = sum;
        return result;
    }
    
    // Handle axis-specific summation for 2D tensors
    if (tensor->ndim == 2) {
        if (axis == 0) {
            // Sum along rows, result shape is [cols]
            int result_shape[] = {tensor->shape[1]};
            CTensor* result = create_tensor(nullptr, result_shape, 1, tensor->device);
            
            for (int j = 0; j < tensor->shape[1]; j++) {
                float sum = 0.0f;
                for (int i = 0; i < tensor->shape[0]; i++) {
                    sum += tensor->data[i * tensor->shape[1] + j];
                }
                result->data[j] = sum;
            }
            return result;
        } else if (axis == 1) {
            // Sum along columns, result shape is [rows]
            int result_shape[] = {tensor->shape[0]};
            CTensor* result = create_tensor(nullptr, result_shape, 1, tensor->device);
            
            for (int i = 0; i < tensor->shape[0]; i++) {
                float sum = 0.0f;
                for (int j = 0; j < tensor->shape[1]; j++) {
                    sum += tensor->data[i * tensor->shape[1] + j];
                }
                result->data[i] = sum;
            }
            return result;
        }
    }
    
    // For other cases, just return a copy for now
    return create_tensor(tensor->data, tensor->shape, tensor->ndim, tensor->device);
}

// Broadcasting helpers
void broadcast_shapes(int* shape1, int ndim1, int* shape2, int ndim2, int* out_shape, int& out_ndim) {
    out_ndim = std::max(ndim1, ndim2);
    for (int i = 0; i < out_ndim; i++) {
        int dim1 = (i < ndim1) ? shape1[ndim1 - 1 - i] : 1;
        int dim2 = (i < ndim2) ? shape2[ndim2 - 1 - i] : 1;
        out_shape[out_ndim - 1 - i] = std::max(dim1, dim2);
    }
}

CTensor* add_broadcasted_tensor(CTensor* a, CTensor* b) {
    // Simplified broadcasting - just returns regular add for now
    return add_tensor(a, b);
}

CTensor* sub_broadcasted_tensor(CTensor* a, CTensor* b) {
    // Simplified broadcasting - just returns regular sub for now
    return sub_tensor(a, b);
}

// Activation functions
CTensor* sigmoid_tensor(CTensor* tensor) {
    CTensor* result = create_tensor(nullptr, tensor->shape, tensor->ndim, tensor->device);
    for (int i = 0; i < tensor->size; i++) {
        result->data[i] = 1.0f / (1.0f + std::exp(-tensor->data[i]));
    }
    return result;
}

CTensor* relu_tensor(CTensor* tensor) {
    CTensor* result = create_tensor(nullptr, tensor->shape, tensor->ndim, tensor->device);
    for (int i = 0; i < tensor->size; i++) {
        result->data[i] = std::max(0.0f, tensor->data[i]);
    }
    return result;
}

CTensor* log_tensor(CTensor* tensor) {
    CTensor* result = create_tensor(nullptr, tensor->shape, tensor->ndim, tensor->device);
    for (int i = 0; i < tensor->size; i++) {
        result->data[i] = std::log(tensor->data[i]);
    }
    return result;
}

CTensor* softmax_tensor(CTensor* tensor, int axis) {
    // Simplified implementation for 1D tensors
    CTensor* result = create_tensor(nullptr, tensor->shape, tensor->ndim, tensor->device);
    
    if (tensor->ndim == 1 || axis == -1) {
        // Find max for numerical stability
        float max_val = *std::max_element(tensor->data, tensor->data + tensor->size);
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < tensor->size; i++) {
            result->data[i] = std::exp(tensor->data[i] - max_val);
            sum += result->data[i];
        }
        
        // Normalize
        for (int i = 0; i < tensor->size; i++) {
            result->data[i] /= sum;
        }
    } else {
        // For multi-dimensional, just copy for now
        std::memcpy(result->data, tensor->data, tensor->size * sizeof(float));
    }
    
    return result;
}

// Distributed computing stub functions
void init_process_group(const char* /* backend */, int /* rank */, int /* world_size */) {
    // No-op stub implementation
    // This would normally initialize MPI or NCCL backend
}

void broadcast_tensor(CTensor* /* tensor */, int /* src */) {
    // No-op stub implementation
    // This would normally broadcast tensor data from src rank to all other ranks
}

void allreduce_sum_tensor(CTensor* /* tensor */) {
    // No-op stub implementation
    // This would normally perform an allreduce sum operation across all ranks
}

} // extern "C"