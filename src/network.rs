use crate::loss::Loss;
use crate::Neuron;
use crate::{activations::Activation::Sigmoid, neuron};
use crate::layer::Intermediary;
use approx_eq::assert_approx_eq;
use crate::linalg_util::outer;
use ndarray::{arr1, Array1, Axis, Array2};

#[derive(Debug)]
pub struct Network {
    pub input_size: usize,
    pub output_size: usize,
    pub hidden_size: usize,
    pub learning_rate: f32,
    pub loss: Loss,
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
                weights: arr1(&[1.0, 1.0]),
                bias: 0.0
            };
            hidden_size
        ];
        let output_layer = vec![
            Neuron {
                activation: Sigmoid,
                weights: arr1(&[1.0, 1.0]),
                bias: 0.0
            };
            output_size
        ];

        let network = Network {
            input_size: input_size,
            output_size: output_size,
            hidden_size: hidden_size,
            learning_rate: learning_rate,
            loss: Loss::MSE,
            _layers: vec![hidden_layer, output_layer],
        };
        return network;
    }

    fn forward(
        &self,
        mut x: Array1<f32>,
        save_intermediaries: bool,
    ) -> (Array1<f32>, Vec<Intermediary>) {
        let mut intermediaries = Vec::new();
        
        let mut first_intermediary = Intermediary::new();
        first_intermediary.o = x.clone();
        intermediaries.push(first_intermediary);

        for layer in &self._layers {
            let mut intermediary = Intermediary::new();
            let mut x_forward_vec: Vec<f32> = Vec::new();
            for neuron in layer.iter() {
                let x_fwd = neuron.forward(&x);
                x_forward_vec.push(x_fwd);
                if save_intermediaries {
                    let z = neuron.activation.inverse(&x_fwd);
                    intermediary.z.push(z);
                }
            }

            x = Array1::from_shape_vec(x_forward_vec.len(), x_forward_vec)
                .expect("Got invalid inputs to forward");
            if save_intermediaries {
                    intermediary.o = x.clone();
                    intermediaries.push(intermediary);
                }
        }

        println!("Intermediaries: {:#?}", intermediaries);
        return (x, intermediaries);
    }
    fn backward(mut self, x: Array1<f32>, y_true: Array1<f32>) {
        let (y_pred, intermediaries) = self.forward(x, true);
        self.update_weights(y_true, y_pred, intermediaries);

    }
    fn update_weights(mut self, y_true: Array1<f32>, y_pred: Array1<f32>, mut intermediaries: Vec<Intermediary>){
        let dis = self.get_dis(&intermediaries, y_true);
        let deltas = self.get_deltas(intermediaries, dis);





    }
    fn get_dis(&self, intermediaries: &Vec<Intermediary>, y_true: Array1<f32>) -> Vec<Array1<f32>>{
        let mut dis: Vec<Array1<f32>> = Vec::new();
        
        // Get the intermediaries in reverse order excluding the first intermediary
        let intermediaries_reversed = intermediaries[1..].iter().rev();
        for (layer_num, intermediary) in intermediaries_reversed.enumerate(){
            if layer_num==0{
                let y_pred = &intermediary.o;
                let dil = self.loss.deriv(&y_true, &y_pred);
                dis.push(arr1(&[dil]));
            }
            else{
                let layer =&self._layers[self._layers.len()-(layer_num)];
                println!("layer: {:#?}", layer);
                for (neuron_num, neuron) in layer.iter().enumerate(){
                    let zl = intermediary.z[neuron_num];
                    let di_l = &dis[dis.len()-1];
                    
                    let di_l_minus_1 = di_l.dot(&neuron.weights.clone().insert_axis(Axis(1)).t())*neuron.activation.deriv(&zl);
                    
                    dis.push(di_l_minus_1);

                }
            }
        }
        return dis;
    }
    fn get_deltas(&self, intermediaries: Vec<Intermediary>, dis: Vec<Array1<f32>>) -> Vec<Array2<f32>>{
        println!("dis: {:?}", dis);
        let mut deltas: Vec<Array2<f32>> = Vec::new();
        let mut delta: Array2<f32>;
        println!("intermediaries: {:?}", intermediaries);
        for (l, di) in dis.iter().rev().enumerate(){
            println!("di: {:?}", di);
            let intermediary = intermediaries[l].o.clone();
            println!("intermediary: {:?}", intermediary.clone().insert_axis(Axis(1)));
            delta = outer(&di, &intermediary);
            println!("delta: {}", delta);
            deltas.push(delta);
        }
        return deltas;

    }
    
}


#[test]
fn test_forward() {
    let mut network = Network::new(2, 2, 1, 0.5);
    let (output, intermediaries) = network.forward(arr1(&[2.0, 3.0]), false);
    assert_eq!(
        output.len(),
        1,
        "testing output size is equal to what was set in initialisation {}, {}",
        output.len(),
        1
    );
    assert_approx_eq!(output[0].into(), 0.8794, 1e-4);
}
#[test]
fn test_forward_intermediaries(){
    let mut network = Network::new(2,2,1,0.5);
    let (output, intermediaries) = network.forward(arr1(&[-2.0, -1.0]), true);
    assert_eq!(intermediaries.len(), network._layers.len())

}

# [test]
fn test_backward(){
    let mut network = Network::new(2,2,1,0.5);
    println!("network: {:#?}", network);
    network.backward(arr1(&[-2.0,-1.0]),arr1(&[1.0]));

}
