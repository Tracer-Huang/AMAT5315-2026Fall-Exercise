use tenferro_cpu::CpuEngine;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;
#[test]
fn tensor_matrix_product_uses_cpu() {
    let a = Tensor::from_vec_col_major([2, 2], vec![1., 3., 2., 4.]).unwrap();
    let b = Tensor::from_vec_col_major([2, 2], vec![2., 1., 0., 2.]).unwrap();
    let c = [&a, &b]
        .einsum("ij,jk->ik", &mut CpuEngine::serial())
        .unwrap();
    assert_eq!(c.as_slice::<f64>().unwrap(), &[4., 10., 4., 8.]);
}
