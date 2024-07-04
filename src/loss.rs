use approx_eq::assert_approx_eq;
use ndarray::{arr1, Array1};
fn mse_loss(y_true: Array1<f32>, y_pred: Array1<f32>) -> f32 {
    (&y_true - &y_pred).map(|v| v.powi(2)).sum() / (y_true.len() as f32)
}

#[test]
fn test_mse_loss_binary() {
    let y_true = arr1(&[1.0, 0.0, 0.0, 1.0]);
    let y_pred = arr1(&[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(mse_loss(y_true, y_pred), 0.5);
}
#[test]
fn test_mse_loss() {
    let y_true = arr1(&[5.0, 4.5, 1.2]);
    let y_pred = arr1(&[5.2, 1.0, 4.5]);
    assert_approx_eq!(mse_loss(y_true, y_pred).into(), 7.726, 1e-4);
}
