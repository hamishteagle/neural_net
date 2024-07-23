use ndarray::{Array1, arr1};

struct Layer {
    input_size: u32,
    output_size: u32,
}
#[derive(Debug, Clone)]
pub struct Intermediary{
    pub o: Array1<f32>,
    pub z: Vec<f32>
}

impl Intermediary{
    pub fn new() -> Intermediary{
        Intermediary{
            o:arr1(&[]),
            z:Vec::new(),
        }
    }
}