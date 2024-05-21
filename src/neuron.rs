use ndarray::Array2;

pub struct Neuron {
    pub weights: Array2<f32>,
    pub bias: f32,
}

impl Neuron {
    fn forward(&self, inputs: &Array2<f32>) -> Array2<f32> {
        // Return inputs*weights + bias
        return inputs.dot(&self.weights) + self.bias;
    }
}
