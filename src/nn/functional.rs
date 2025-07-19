use crate::tensor::Tensor;

pub fn sigmoid(input: &Tensor) -> Tensor {
    input.sigmoid()
}

pub fn softmax(input: &Tensor, dim: i32) -> Tensor {
    input.softmax(dim)
}

pub fn relu(input: &Tensor) -> Tensor {
    input.relu()
}

pub fn log_softmax(input: &Tensor, dim: i32) -> Tensor {
    let softmax_result = input.softmax(dim);
    softmax_result.log()
}

pub fn mse_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let diff = input.sub(target);
    let squared = diff.elementwise_mul(&diff);
    let sum = squared.sum(-1, false);
    sum.scalar_mul(1.0 / input.numel as f32)
}

pub fn cross_entropy_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let log_probs = log_softmax(input, -1);
    let loss = target.elementwise_mul(&log_probs);
    let neg_loss = loss.scalar_mul(-1.0);
    neg_loss.sum(-1, false)
}