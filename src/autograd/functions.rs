// Autograd functions for backpropagation - implements gradient computation for various operations
// This file contains backward pass implementations for all tensor operations that support automatic differentiation
// Connected to: src/tensor.rs (GradFn trait), src/nn/ (neural network layer backward passes)
// Used by: Training loops, optimizer updates, and any tensor operations that require gradients

pub use self::ops::*;

use ndarray::{ArrayD, Axis};

/// Trait implemented by every backward operation node.
pub trait Backward {
    /// Given the gradient flowing from the next layer, return the gradients with respect to this operation's inputs.
    fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>>;
}

mod ops {
    use super::*;
    use ndarray::{IxDyn, Ix2};

    /// Back-prop for a simple element-wise addition (`x + y`) without broadcasting.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct AddBackward;

    impl Backward for AddBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            vec![gradient.clone(), gradient.clone()]
        }
    }

    /// Back-prop for addition that involved broadcasting.
    #[derive(Debug, Clone)]
    pub struct AddBroadcastedBackward {
        pub x_shape: Vec<usize>,
        pub y_shape: Vec<usize>,
    }

    impl AddBroadcastedBackward {
        fn reshape_gradient(mut gradient: ArrayD<f32>, target_shape: &[usize]) -> ArrayD<f32> {
            // 1. Collapse leading broadcasted dimensions.
            while gradient.ndim() > target_shape.len() {
                gradient = gradient.sum_axis(Axis(0));
            }
            // 2. For axes where the target dim is 1, sum across that axis (keepdim).
            for (axis_idx, &dim) in target_shape.iter().enumerate() {
                if dim == 1 {
                    gradient = gradient.sum_axis(Axis(axis_idx)).insert_axis(Axis(axis_idx));
                }
            }
            gradient
        }
    }

    impl Backward for AddBroadcastedBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            let grad_x = Self::reshape_gradient(gradient.clone(), &self.x_shape);
            let grad_y = Self::reshape_gradient(gradient.clone(), &self.y_shape);
            vec![grad_x, grad_y]
        }
    }

    /// Back-prop for subtraction (`x - y`) without broadcasting.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct SubBackward;

    impl Backward for SubBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            vec![gradient.clone(), -gradient]
        }
    }

    /// Back-prop for broadcasted subtraction.
    #[derive(Debug, Clone)]
    pub struct SubBroadcastedBackward {
        pub x_shape: Vec<usize>,
        pub y_shape: Vec<usize>,
    }

    impl SubBroadcastedBackward {
        fn reshape_gradient(mut gradient: ArrayD<f32>, target_shape: &[usize]) -> ArrayD<f32> {
            while gradient.ndim() > target_shape.len() {
                gradient = gradient.sum_axis(Axis(0));
            }
            for (axis_idx, &dim) in target_shape.iter().enumerate() {
                if dim == 1 {
                    gradient = gradient.sum_axis(Axis(axis_idx)).insert_axis(Axis(axis_idx));
                }
            }
            gradient
        }
    }

    impl Backward for SubBroadcastedBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            let grad_x = Self::reshape_gradient(gradient.clone(), &self.x_shape);
            let grad_y = Self::reshape_gradient(gradient.clone(), &self.y_shape);
            vec![grad_x, -grad_y]
        }
    }

    /// Back-prop for scalar multiplication.
    #[derive(Debug, Clone)]
    pub struct ScalarMulBackward {
        pub scalar: f32,
    }

    impl Backward for ScalarMulBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            vec![gradient * self.scalar]
        }
    }

    /// Back-prop for element-wise multiplication.
    #[derive(Debug, Clone)]
    pub struct ElementwiseMulBackward {
        pub x: ArrayD<f32>,
        pub y: ArrayD<f32>,
    }

    impl Backward for ElementwiseMulBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            vec![&self.y * gradient, &self.x * gradient]
        }
    }

    /// Back-prop for matrix multiplication.
    #[derive(Debug, Clone)]
    pub struct MatmulBackward {
        pub x: ArrayD<f32>,
        pub y: ArrayD<f32>,
    }

    impl Backward for MatmulBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            let x_2d = self.x.view().into_dimensionality::<Ix2>().unwrap();
            let y_2d = self.y.view().into_dimensionality::<Ix2>().unwrap();
            let grad_2d = gradient.view().into_dimensionality::<Ix2>().unwrap();

            if self.x.ndim() != self.y.ndim() {
                // Broadcasted case
                let aux = grad_2d.dot(&y_2d.t());
                let aux_sum = aux.sum_axis(Axis(0));
                let grad_x = aux_sum.into_dyn();
                let grad_y = x_2d.t().dot(&grad_2d).into_dyn();
                vec![grad_x, grad_y]
            } else {
                let grad_x = grad_2d.dot(&y_2d.t()).into_dyn();
                let grad_y = x_2d.t().dot(&grad_2d).into_dyn();
                vec![grad_x, grad_y]
            }
        }
    }

    /// Back-prop for power operation.
    #[derive(Debug, Clone)]
    pub struct PowBackward {
        pub base: ArrayD<f32>,
        pub exponent: ArrayD<f32>,
    }

    impl Backward for PowBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            // Element-wise gradient for base: gradient * exponent * base^(exponent-1)
            let mut grad_base = gradient.clone();
            for ((exp_val, base_val), grad_val) in self.exponent.iter().zip(self.base.iter()).zip(grad_base.iter_mut()) {
                *grad_val = *grad_val * exp_val * base_val.powf(exp_val - 1.0);
            }
            
            // Element-wise gradient for exponent: gradient * base^exponent * ln(base)
            let mut grad_exponent = gradient.clone();
            for ((exp_val, base_val), grad_val) in self.exponent.iter().zip(self.base.iter()).zip(grad_exponent.iter_mut()) {
                *grad_val = *grad_val * base_val.powf(*exp_val) * base_val.ln();
            }
            
            vec![grad_base, grad_exponent]
        }
    }

    /// Back-prop for logarithm.
    #[derive(Debug, Clone)]
    pub struct LogBackward {
        pub x: ArrayD<f32>,
    }

    impl Backward for LogBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            vec![gradient / &self.x]
        }
    }

    /// Back-prop for sum operation.
    #[derive(Debug, Clone)]
    pub struct SumBackward {
        pub input_shape: Vec<usize>,
        pub axis: Option<usize>,
        pub keepdim: bool,
    }

    impl Backward for SumBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            if self.axis.is_none() {
                // Sum all elements to scalar
                let scalar_grad = gradient[vec![0; gradient.ndim()].as_slice()];
                let grad_output = ArrayD::from_elem(IxDyn(&self.input_shape), scalar_grad);
                vec![grad_output]
            } else {
                let axis = self.axis.unwrap();
                let mut grad_shape = self.input_shape.clone();
                
                if self.keepdim {
                    grad_shape[axis] = 1;
                } else {
                    grad_shape.remove(axis);
                }
                
                let reshaped_grad = gradient.clone().into_shape(IxDyn(&grad_shape)).unwrap();
                let broadcasted = reshaped_grad.broadcast(IxDyn(&self.input_shape)).unwrap().to_owned();
                vec![broadcasted]
            }
        }
    }

    /// Back-prop for reshape operation.
    #[derive(Debug, Clone)]
    pub struct ReshapeBackward {
        pub input_shape: Vec<usize>,
    }

    impl Backward for ReshapeBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            vec![gradient.clone().into_shape(IxDyn(&self.input_shape)).unwrap()]
        }
    }

    /// Back-prop for transpose operation.
    #[derive(Debug, Clone)]
    pub struct TransposeBackward {
        pub axis1: usize,
        pub axis2: usize,
    }

    impl Backward for TransposeBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            let mut grad = gradient.clone();
            grad.swap_axes(self.axis1, self.axis2);
            vec![grad]
        }
    }

    /// Back-prop for matrix transpose (.T).
    #[derive(Debug, Default, Clone, Copy)]
    pub struct TBackward;

    impl Backward for TBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            let grad_2d = gradient.view().into_dimensionality::<Ix2>().unwrap();
            vec![grad_2d.t().to_owned().into_dyn()]
        }
    }

    /// Back-prop for division.
    #[derive(Debug, Clone)]
    pub struct DivisionBackward {
        pub x: ArrayD<f32>,
        pub y: ArrayD<f32>,
    }

    impl Backward for DivisionBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            let grad_x = gradient / &self.y;
            let grad_y = -gradient * &self.x / (&self.y * &self.y);
            vec![grad_x, grad_y]
        }
    }

    /// Back-prop for sine.
    #[derive(Debug, Clone)]
    pub struct SinBackward {
        pub x: ArrayD<f32>,
    }

    impl Backward for SinBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            vec![gradient * self.x.mapv(|x| x.cos())]
        }
    }

    /// Back-prop for cosine.
    #[derive(Debug, Clone)]
    pub struct CosBackward {
        pub x: ArrayD<f32>,
    }

    impl Backward for CosBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            vec![-gradient * self.x.mapv(|x| x.sin())]
        }
    }

    /// Back-prop for max operation.
    #[derive(Debug, Clone)]
    pub struct MaxBackward {
        pub input: ArrayD<f32>,
        pub axis: Option<usize>,
        pub keepdim: bool,
    }

    impl Backward for MaxBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            if self.axis.is_none() {
                // Global max
                let max_val = self.input.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                let mask = self.input.mapv(|x| if (x - max_val).abs() < f32::EPSILON { 1.0 } else { 0.0 });
                let mask_sum = mask.sum();
                let scalar_grad = gradient[vec![0; gradient.ndim()].as_slice()];
                let grad_output = mask * (scalar_grad / mask_sum);
                vec![grad_output]
            } else {
                let axis = self.axis.unwrap();
                let max_vals = self.input.fold_axis(Axis(axis), f32::NEG_INFINITY, |acc, &x| acc.max(x));
                let max_vals_expanded = if self.keepdim {
                    max_vals.insert_axis(Axis(axis))
                } else {
                    max_vals.insert_axis(Axis(axis))
                };
                let mask = self.input.mapv(|x| x) - max_vals_expanded.broadcast(self.input.raw_dim()).unwrap();
                let mask = mask.mapv(|x| if x.abs() < f32::EPSILON { 1.0 } else { 0.0 });
                
                let grad_expanded = if self.keepdim {
                    gradient.clone()
                } else {
                    gradient.clone().insert_axis(Axis(axis))
                };
                let grad_broadcasted = grad_expanded.broadcast(self.input.raw_dim()).unwrap();
                vec![grad_broadcasted.to_owned() * mask]
            }
        }
    }

    /// Back-prop for min operation.
    #[derive(Debug, Clone)]
    pub struct MinBackward {
        pub input: ArrayD<f32>,
        pub axis: Option<usize>,
        pub keepdim: bool,
    }

    impl Backward for MinBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            if self.axis.is_none() {
                // Global min
                let min_val = self.input.fold(f32::INFINITY, |acc, &x| acc.min(x));
                let mask = self.input.mapv(|x| if (x - min_val).abs() < f32::EPSILON { 1.0 } else { 0.0 });
                let mask_sum = mask.sum();
                let scalar_grad = gradient[vec![0; gradient.ndim()].as_slice()];
                let grad_output = mask * (scalar_grad / mask_sum);
                vec![grad_output]
            } else {
                let axis = self.axis.unwrap();
                let min_vals = self.input.fold_axis(Axis(axis), f32::INFINITY, |acc, &x| acc.min(x));
                let min_vals_expanded = if self.keepdim {
                    min_vals.insert_axis(Axis(axis))
                } else {
                    min_vals.insert_axis(Axis(axis))
                };
                let mask = self.input.mapv(|x| x) - min_vals_expanded.broadcast(self.input.raw_dim()).unwrap();
                let mask = mask.mapv(|x| if x.abs() < f32::EPSILON { 1.0 } else { 0.0 });
                
                let grad_expanded = if self.keepdim {
                    gradient.clone()
                } else {
                    gradient.clone().insert_axis(Axis(axis))
                };
                let grad_broadcasted = grad_expanded.broadcast(self.input.raw_dim()).unwrap();
                vec![grad_broadcasted.to_owned() * mask]
            }
        }
    }

    /// Back-prop for cross entropy loss.
    #[derive(Debug, Clone)]
    pub struct CrossEntropyLossBackward {
        pub logits: ArrayD<f32>,
        pub targets: ArrayD<f32>,
    }

    impl Backward for CrossEntropyLossBackward {
        fn backward(&self, _gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            let softmax = if self.logits.ndim() == 1 {
                // 1D case
                let max_val = self.logits.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                let exp_logits = self.logits.mapv(|x| (x - max_val).exp());
                let sum_exp = exp_logits.sum();
                exp_logits / sum_exp
            } else {
                // 2D batched case
                let batch_size = self.logits.shape()[0];
                let mut softmax = ArrayD::zeros(self.logits.raw_dim());
                for i in 0..batch_size {
                    let row = self.logits.index_axis(Axis(0), i);
                    let max_val = row.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                    let exp_row = row.mapv(|x| (x - max_val).exp());
                    let sum_exp = exp_row.sum();
                    let softmax_row = exp_row / sum_exp;
                    softmax.index_axis_mut(Axis(0), i).assign(&softmax_row);
                }
                softmax
            };

            let grad_logits = if self.logits.ndim() == 1 {
                softmax - &self.targets
            } else {
                let batch_size = self.logits.shape()[0] as f32;
                (softmax - &self.targets) / batch_size
            };

            vec![grad_logits, ArrayD::zeros(self.targets.raw_dim())] // targets have no gradient
        }
    }

    /// Back-prop for sigmoid.
    #[derive(Debug, Clone)]
    pub struct SigmoidBackward {
        pub input: ArrayD<f32>,
    }

    impl Backward for SigmoidBackward {
        fn backward(&self, gradient: &ArrayD<f32>) -> Vec<ArrayD<f32>> {
            let sigmoid_x = self.input.mapv(|x| 1.0 / (1.0 + (-x).exp()));
            let grad_input = gradient * &sigmoid_x * sigmoid_x.mapv(|x| 1.0 - x);
            vec![grad_input]
        }
    }
} 