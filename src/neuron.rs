use ndarray::{arr1, Array1};

pub struct Neuron {
    weights: Array1<f32>,
    bias: f32,
}

impl Default for Neuron {
    fn default() -> Self {
        Neuron {
            weights: arr1(&[0.5, 0.5]),
            bias: 0.5,
        }
    }
}

impl Neuron {
    fn forward(&self, inputs: &Array1<f32>) -> f32 {
        // Return inputs*weights + bias
        return self.weights.dot(inputs) + self.bias;
    }
}
