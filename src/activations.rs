#[derive(Clone, Debug)]
pub enum Activation {
    Sigmoid,
}

impl Activation {
    pub fn call(&self, x: &f32) -> f32 {
        match self {
            Activation::Sigmoid => Sigmoid.call(x),
        }
    }
    pub fn deriv(&self, x: &f32) -> f32 {
        match self {
            Activation::Sigmoid => Sigmoid.deriv(x),
        }
    }
}
pub struct Sigmoid;

impl Sigmoid {
    fn call(&self, x: &f32) -> f32 {
        return 1.0 / (1.0 + (-x).exp());
    }
    fn deriv(&self, x: &f32) -> f32 {
        return x * (1.0 - x);
    }
}
