use crate::tensor::Tensor;

pub trait Transform {
    fn apply(&self, input: Tensor) -> Tensor;
}

pub struct ToTensor;

impl Transform for ToTensor {
    fn apply(&self, input: Tensor) -> Tensor {
        input
    }
}

pub struct Reshape {
    shape: Vec<usize>,
}

impl Reshape {
    pub fn new(shape: Vec<usize>) -> Self {
        Self { shape }
    }
}

impl Transform for Reshape {
    fn apply(&self, input: Tensor) -> Tensor {
        input.reshape(&self.shape)
    }
}

pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }
}

impl Transform for Compose {
    fn apply(&self, input: Tensor) -> Tensor {
        self.transforms.iter().fold(input, |acc, transform| {
            transform.apply(acc)
        })
    }
}
