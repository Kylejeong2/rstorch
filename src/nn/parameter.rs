use crate::tensor::Tensor;

pub struct ParameterTensor {
    base: Tensor,
}

impl ParameterTensor {
    pub fn new(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        // Generate random data as a flat vector
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let vec_data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let mut tensor = Tensor::from_vec_device(vec_data, shape, "cpu");
        tensor.requires_grad = true;
        Self {
            base: tensor,
        }
    }
    
    pub fn from_tensor(tensor: Tensor) -> Self {
        Self { base: tensor }
    }
}

impl crate::nn::module::Parameter for ParameterTensor {
    fn zero_grad(&mut self) {
        self.base.grad = None;
    }
    
    fn requires_grad(&self) -> bool {
        self.base.requires_grad
    }
    
    fn set_requires_grad(&mut self, requires_grad: bool) {
        self.base.requires_grad = requires_grad;
    }
    
    fn to_device(&mut self, device: &str) {
        self.base.device = device.to_string();
    }
    
    fn shape(&self) -> &[usize] {
        &self.base.shape
    }
    
    fn data(&self) -> &Tensor {
        &self.base
    }
    
    fn data_mut(&mut self) -> &mut Tensor {
        &mut self.base
    }
}