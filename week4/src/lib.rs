use rustfft::{num_complex::Complex64, Fft, FftPlanner};
use std::sync::Arc;
pub type Spectrum = Vec<Complex64>;
pub struct Spectral {
    pub n: usize,
    forward_fft: Arc<dyn Fft<f64>>,
    inverse_fft: Arc<dyn Fft<f64>>,
}
impl Spectral {
    pub fn new(n: usize) -> Self {
        assert!(
            n >= 4 && n.is_power_of_two(),
            "n must be a power of two >= 4"
        );
        let mut planner = FftPlanner::new();
        Self {
            n,
            forward_fft: planner.plan_fft_forward(n),
            inverse_fft: planner.plan_fft_inverse(n),
        }
    }
    fn transform(&self, mut a: Spectrum, inverse: bool) -> Spectrum {
        assert_eq!(a.len(), self.n * self.n);
        let fft = if inverse {
            &self.inverse_fft
        } else {
            &self.forward_fft
        };
        let mut scratch = vec![Complex64::default(); fft.get_inplace_scratch_len()];
        for row in a.chunks_exact_mut(self.n) {
            fft.process_with_scratch(row, &mut scratch);
        }
        let mut col = vec![Complex64::default(); self.n];
        for x in 0..self.n {
            for y in 0..self.n {
                col[y] = a[y * self.n + x];
            }
            fft.process_with_scratch(&mut col, &mut scratch);
            for y in 0..self.n {
                a[y * self.n + x] = col[y];
            }
        }
        // RustFFT is unnormalised in both directions: one 1/N per inverse axis.
        if inverse {
            for z in &mut a {
                *z /= (self.n * self.n) as f64;
            }
        }
        a
    }
    pub fn forward(&self, f: &[f64]) -> Spectrum {
        self.transform(f.iter().map(|&x| Complex64::new(x, 0.)).collect(), false)
    }
    pub fn inverse(&self, f: &[Complex64]) -> Vec<f64> {
        self.transform(f.to_vec(), true)
            .iter()
            .map(|x| x.re)
            .collect()
    }
    pub fn wave(&self, j: usize) -> f64 {
        if j < self.n / 2 {
            j as f64
        } else {
            j as f64 - self.n as f64
        }
    }
    fn derivative_wave(&self, j: usize) -> f64 {
        if j == self.n / 2 {
            0.
        } else {
            self.wave(j)
        }
    }
    pub fn k2(&self, i: usize) -> f64 {
        self.wave(i % self.n).powi(2) + self.wave(i / self.n).powi(2)
    }
    pub fn differentiate_hat(&self, f: &[Complex64], axis: usize) -> Spectrum {
        assert!(axis < 2);
        f.iter()
            .enumerate()
            .map(|(i, &z)| {
                z * Complex64::new(
                    0.,
                    self.derivative_wave(if axis == 0 { i % self.n } else { i / self.n }),
                )
            })
            .collect()
    }
    pub fn derivative(&self, f: &[f64], axis: usize) -> Vec<f64> {
        self.inverse(&self.differentiate_hat(&self.forward(f), axis))
    }
    pub fn poisson_hat(&self, w: &[Complex64]) -> Spectrum {
        w.iter()
            .enumerate()
            .map(|(i, &z)| {
                if i == 0 {
                    Complex64::default()
                } else {
                    z / self.k2(i)
                }
            })
            .collect()
    }
    pub fn poisson(&self, w: &[f64]) -> Vec<f64> {
        self.inverse(&self.poisson_hat(&self.forward(w)))
    }
    pub fn velocity(&self, w: &[Complex64]) -> (Vec<f64>, Vec<f64>) {
        let psi = self.poisson_hat(w);
        let u = self.inverse(&self.differentiate_hat(&psi, 1));
        let v = self
            .inverse(&self.differentiate_hat(&psi, 0))
            .into_iter()
            .map(|x| -x)
            .collect();
        (u, v)
    }
    pub fn filter(&self, w: &mut [Complex64]) {
        for (i, z) in w.iter_mut().enumerate() {
            if self.wave(i % self.n).abs() > (self.n / 3) as f64
                || self.wave(i / self.n).abs() > (self.n / 3) as f64
            {
                *z = Complex64::default();
            }
        }
    }
}

pub mod flow;
pub mod io;
