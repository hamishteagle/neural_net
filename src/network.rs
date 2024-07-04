use crate::activations::Activation::Sigmoid;
use crate::Neuron;
use approx_eq::assert_approx_eq;
use ndarray::{arr1, Array1};

pub struct Network {
    pub input_size: usize,
    pub output_size: usize,
    pub hidden_size: usize,
    pub learning_rate: f32,
    _weights: Array1<f32>,
    _bias: f32,
    _layers: Vec<Vec<Neuron>>,
}

impl Network {
    fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        learning_rate: f32,
    ) -> Network {
        //TODO: do something with input size
        let hidden_layer = vec![
            Neuron {
                activation: Sigmoid,
                weights: arr1(&[0.0, 1.0]),
                bias: 0.0
            };
            hidden_size
        ];
        let output_layer = vec![
            Neuron {
                activation: Sigmoid,
                weights: arr1(&[0.0, 1.0]),
                bias: 0.0
            };
            output_size
        ];

        let network = Network {
            input_size: input_size,
            output_size: output_size,
            hidden_size: hidden_size,
            learning_rate: learning_rate,
            _weights: arr1(&[0.0, 1.0]),
            _bias: 0.0,
            _layers: vec![hidden_layer, output_layer],
        };
        return network;
    }

    fn forward(mut self, mut x: Array1<f32>) -> Array1<f32> {
        for layer in &mut self._layers {
            let mut x_forward_vec: Vec<f32> = Vec::new();
            for neuron in layer.iter_mut() {
                x_forward_vec.push(neuron.forward(&x));
                println!("x_forward_vec: {:?}", x_forward_vec);
            }
            x = Array1::from_shape_vec(x_forward_vec.len(), x_forward_vec)
                .expect("Got invalid inputs to forward")
        }

        return x;
    }
    fn backward(self, x: Array1<f32>) {}
}

#[test]
fn test_forward() {
    let network = Network::new(2, 2, 1, 0.5);
    let output = network.forward(arr1(&[2.0, 3.0]));
    assert_eq!(
        output.len(),
        1,
        "testing output size is equal to what was set in initialisation {}, {}",
        output.len(),
        1
    );
    assert_approx_eq!(output[0].into(), 0.7216, 1e-4);
}
