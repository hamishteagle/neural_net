use approx_eq::assert_approx_eq;
use ndarray::Array1;

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
    pub fn inverse(&self, x: &f32) -> f32 {
        match self {
            Activation::Sigmoid => Sigmoid.inverse(x),
        }
    }
}
pub struct Sigmoid;

impl Sigmoid {
    fn call(&self, x: &f32) -> f32 {
        return 1.0 / (1.0 + (-x).exp());
    }
    fn deriv(&self, x: &f32) -> f32 {
        return self.call(x) * (1.0 - self.call(x));
    }
    fn inverse(&self, x: &f32) -> f32 {
        return (x / (1.0 - x)).ln();
    }
}


#[test]
fn test_deriv_sigmoid(){
    assert_approx_eq!(Sigmoid.deriv(&3.0).into(), 0.04518, 1e-4);
}