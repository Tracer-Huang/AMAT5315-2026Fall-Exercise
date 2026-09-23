use crate::{Spectral, Spectrum};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rustfft::num_complex::Complex64;
use std::f64::consts::TAU;
pub fn taylor_green(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut u = Vec::with_capacity(n * n);
    let mut v = Vec::with_capacity(n * n);
    for y in 0..n {
        for x in 0..n {
            let (x, y) = (TAU * x as f64 / n as f64, TAU * y as f64 / n as f64);
            u.push(x.cos() * y.sin());
            v.push(-x.sin() * y.cos());
        }
    }
    (u, v)
}
pub fn random_field(n: usize, seed: u64, lo: usize, hi: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(lo >= 1 && lo <= hi && hi <= n / 3);
    let s = Spectral::new(n);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut w = vec![Complex64::default(); n * n];
    // Enumerate physical modes, not array indices: phases are independent of N.
    // Only one half-plane is drawn; its conjugate ensures a real-valued field.
    for ky in 0..=hi as i32 {
        for kx in -(hi as i32)..=hi as i32 {
            if ky == 0 && kx <= 0 {
                continue;
            }
            let k2 = (kx * kx + ky * ky) as usize;
            if k2 < lo * lo || k2 > hi * hi {
                continue;
            }
            let phase = rng.gen::<f64>() * TAU;
            let z = Complex64::from_polar((n * n) as f64, phase);
            let index = |x: i32, y: i32| {
                y.rem_euclid(n as i32) as usize * n + x.rem_euclid(n as i32) as usize
            };
            w[index(kx, ky)] = z;
            w[index(-kx, -ky)] = z.conj();
        }
    }
    let (mut u, mut v) = s.velocity(&w);
    let scale = (0.5 / energy(&u, &v)).sqrt();
    for a in u.iter_mut().chain(&mut v) {
        *a *= scale;
    }
    (u, v)
}
pub fn energy(u: &[f64], v: &[f64]) -> f64 {
    u.iter().zip(v).map(|(a, b)| a * a + b * b).sum::<f64>() / (2. * u.len() as f64)
}
pub fn vorticity(s: &Spectral, u: &[f64], v: &[f64]) -> Spectrum {
    let vx = s.derivative(v, 0);
    let uy = s.derivative(u, 1);
    s.forward(&vx.iter().zip(uy).map(|(a, b)| a - b).collect::<Vec<_>>())
}
fn rhs(s: &Spectral, w: &[Complex64], p: &[[f64; 2]], nu: f64) -> (Spectrum, Vec<[f64; 2]>) {
    let mut clean = w.to_vec();
    s.filter(&mut clean);
    let (u, v) = s.velocity(&clean);
    let wx = s.inverse(&s.differentiate_hat(&clean, 0));
    let wy = s.inverse(&s.differentiate_hat(&clean, 1));
    let adv: Vec<_> = u
        .iter()
        .zip(&v)
        .zip(wx.iter().zip(wy))
        .map(|((u, v), (x, y))| -(u * x + v * y))
        .collect();
    let mut rate = s.forward(&adv);
    s.filter(&mut rate);
    for (i, z) in rate.iter_mut().enumerate() {
        *z -= nu * s.k2(i) * clean[i];
    }
    let prate = p.iter().map(|&q| interpolate(s, &u, &v, q)).collect();
    (rate, prate)
}
pub fn step(s: &Spectral, w: &mut Spectrum, p: &mut [[f64; 2]], nu: f64, dt: f64) {
    let ws =
        |r: &[Complex64], h: f64| -> Spectrum { w.iter().zip(r).map(|(a, b)| a + h * b).collect() };
    let ps = |r: &[[f64; 2]], h: f64| -> Vec<[f64; 2]> {
        p.iter()
            .zip(r)
            .map(|(a, b)| [a[0] + h * b[0], a[1] + h * b[1]])
            .collect()
    };
    let (k1, q1) = rhs(s, w, p, nu);
    let (k2, q2) = rhs(s, &ws(&k1, dt / 2.), &ps(&q1, dt / 2.), nu);
    let (k3, q3) = rhs(s, &ws(&k2, dt / 2.), &ps(&q2, dt / 2.), nu);
    let (k4, q4) = rhs(s, &ws(&k3, dt), &ps(&q3, dt), nu);
    for i in 0..w.len() {
        w[i] += dt / 6. * (k1[i] + 2. * k2[i] + 2. * k3[i] + k4[i]);
    }
    s.filter(w);
    // Use each RK stage's velocity at that stage's particle position.
    for i in 0..p.len() {
        for d in 0..2 {
            p[i][d] = (p[i][d] + dt / 6. * (q1[i][d] + 2. * q2[i][d] + 2. * q3[i][d] + q4[i][d]))
                .rem_euclid(TAU);
        }
    }
}
pub fn interpolate(s: &Spectral, u: &[f64], v: &[f64], p: [f64; 2]) -> [f64; 2] {
    let gx = p[0].rem_euclid(TAU) * s.n as f64 / TAU;
    let gy = p[1].rem_euclid(TAU) * s.n as f64 / TAU;
    let ix = gx.floor() as usize % s.n;
    let iy = gy.floor() as usize % s.n;
    let a = gx - gx.floor();
    let b = gy - gy.floor();
    let at = |f: &[f64]| {
        (1. - b) * ((1. - a) * f[iy * s.n + ix] + a * f[iy * s.n + (ix + 1) % s.n])
            + b * ((1. - a) * f[((iy + 1) % s.n) * s.n + ix]
                + a * f[((iy + 1) % s.n) * s.n + (ix + 1) % s.n])
    };
    [at(u), at(v)]
}
