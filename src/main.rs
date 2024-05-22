//use crate::Neuron;

mod neuron;
use crate::neuron::Neuron;
use ::ndarray::arr2;

use std::io;
fn main() {
    let mut neuron = Neuron::default();
}

fn get_input_size() -> u32 {
    loop {
        println!("Input the size of the input (integer)");
        let mut input_size = String::new();

        io::stdin()
            .read_line(&mut input_size)
            .expect("Failed to read input size");

        let input_size: u32 = match input_size.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                println!("Couldn't understand input {input_size}, please try again");
                continue;
            }
        };
        println!("Got input size {input_size}");
        return input_size;
    }
}
