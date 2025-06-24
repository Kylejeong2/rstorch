use rstorch::Tensor;

#[test]
fn test_add() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_vec(vec![4.0, 3.0, 2.0, 1.0], &[2, 2]);
    let c = &a + &b;
    assert_eq!(c.data.as_slice().unwrap(), &[5.0, 5.0, 5.0, 5.0]);
}

#[test]
fn test_matmul() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_vec(vec![4.0, 3.0, 2.0, 1.0], &[2, 2]);
    let c = a.matmul(&b);
    assert_eq!(c.data.as_slice().unwrap(), &[8.0, 5.0, 20.0, 13.0]);
}

#[test]
fn test_sum() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let s = a.sum();
    assert_eq!(s.data.as_slice().unwrap(), &[10.0]);
} 