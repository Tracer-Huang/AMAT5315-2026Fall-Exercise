use week10::svd;
#[test]
fn reconstruction_of_rectangular_matrix() {
 let a = vec![1.,2.,3.,4.,5.,6.];
 let (u,s,vt) = svd(&a,3,2);
 for j in 0..2 { for i in 0..3 {
  let v=(0..2).map(|k|u[i+3*k]*s[k]*vt[k+2*j]).sum::<f64>();
  assert!((v-a[i+3*j]).abs()<1e-10);
 }}
}
