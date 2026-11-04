use tenferro_cpu::CpuEngine;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;
#[test]
fn ising_ring_contraction_equals_eight_explicit_states() {
    let spins = [-1.0_f64, 1.0_f64];
    let mut bond = vec![];
    for &b in &spins {
        for &a in &spins {
            bond.push((0.5 * a * b).exp())
        }
    }
    let t = Tensor::from_vec_col_major([2, 2], bond).unwrap();
    let z = [&t, &t, &t]
        .einsum("ab,bc,ca->", &mut CpuEngine::serial())
        .unwrap()
        .as_slice::<f64>()
        .unwrap()[0];
    let mut exact = 0.;
    for a in spins {
        for b in spins {
            for c in spins {
                exact += (0.5 * (a * b + b * c + c * a)).exp()
            }
        }
    }
    assert!((z - exact).abs() < 1e-12);
    assert!((z - 12.60256209895193).abs() < 1e-12);
}
