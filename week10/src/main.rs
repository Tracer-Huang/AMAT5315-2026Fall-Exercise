use rand::{SeedableRng,Rng};
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;
use std::{fs::File,io::{Write,BufWriter}};
fn save(path:&str,a:&[f64]) {let mut f=BufWriter::new(File::create(path).unwrap()); for x in a { f.write_all(&x.to_le_bytes()).unwrap(); }}
fn main(){let n=512;let mut rng=ChaCha8Rng::seed_from_u64(2026);let a:Vec<f64>=(0..n*n).map(|_|rng.sample(StandardNormal)).collect();let (u,s,vt)=week10::svd(&a,n,n);std::fs::create_dir_all("artifacts/svd").unwrap();save("artifacts/svd/matrix.f64",&a);save("artifacts/svd/singular.f64",&s);for rank in [16,64,128,256,512]{let mut b=vec![0.;n*n];for j in 0..n{for k in 0..rank {let v=s[k]*vt[k+n*j];for i in 0..n{b[i+n*j]+=u[i+n*k]*v;}}}save(&format!("artifacts/svd/rank{rank}.f64"),&b);}}
