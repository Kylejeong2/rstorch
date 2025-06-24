use ndarray::{ArrayD, IxDyn, Ix2, ArrayView2};
use std::ops::Add;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: ArrayD<f32>,
}

impl Tensor {
    /// Create a new tensor from raw data and shape.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        Self {
            data: ArrayD::from_shape_vec(IxDyn(shape), data).expect("Invalid shape"),
        }
    }

    /// Return a tensor filled with zeros of given shape
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::zeros(IxDyn(shape)),
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.data.shape(), other.data.shape(), "Shape mismatch in add");
        let result = &self.data + &other.data;
        Self { data: result }
    }

    /// Matrix multiplication (2-D tensors)
    pub fn matmul(&self, other: &Self) -> Self {
        let a: ArrayView2<'_, f32> = self
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("matmul requires 2-D tensors");
        let b: ArrayView2<'_, f32> = other
            .data
            .view()
            .into_dimensionality::<Ix2>()
            .expect("matmul requires 2-D tensors");

        let res = a.dot(&b);
        Self {
            data: res.into_dyn(),
        }
    }

    /// Sum of all elements producing a scalar tensor
    pub fn sum(&self) -> Self {
        let s = self.data.sum();
        Self { data: ArrayD::from_elem(IxDyn(&[]), s) }
    }

    /// Get shape as slice
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }
}

impl Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
} 