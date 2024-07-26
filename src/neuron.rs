use crate::activations::Activation;
use ndarray::{arr1, Array1};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Neuron {
    pub activation: Activation,
    pub weights: Array1<f32>,
    pub bias: f32,
}

impl Default for Neuron {
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        let weights = arr1(&[rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)]);
        let bias = rng.gen_range(-1.0..1.0);
        Neuron {
            activation: Activation::Sigmoid,
            weights: weights,
            bias: bias,
        }
    }
}

impl Neuron {
    pub fn forward(&self, inputs: &Array1<f32>) -> f32 {
        // Return inputs*weights + bias
        let w_b = self.weights.dot(inputs) + self.bias;
        return self.activation.call(&w_b);
    }
}

#[test]
fn test_forward_sigmoid() {
    let neuron = Neuron::default();
    let input = arr1(&[2.0, 3.0]);
    let output = neuron.forward(&input);
    assert_eq!(output, 0.999089);
}
