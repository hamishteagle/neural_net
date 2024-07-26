use ndarray::{Array1, Axis, Array2, arr1, arr2};

pub fn outer(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32>{
    let a_t = a.clone().insert_axis(Axis(1));
    let b_t = b.clone().insert_axis(Axis(1));
    a_t.dot(&b_t.t())
}


#[test]
fn test_outer(){
    let A = arr1(&[1.0, 0.0]);
    let B = arr1(&[1.0, 1.0]);
    let outer_result = outer(&A,&B);
    assert_eq!(outer_result, arr2(&[[1.0,1.0],[0.0,0.0]]))
}
#[test]
fn test_outer_shapes(){
    let A = arr1(&[1.0]);
    let B = arr1(&[1.0, 0.0]);
    assert_eq!(outer(&A,&B), arr2(&[[1.0, 0.0]]))
}