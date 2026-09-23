use std::f64::consts::TAU;
use week4_fluid::Spectral;

fn grid(n: usize, f: impl Fn(f64, f64) -> f64) -> Vec<f64> {
    (0..n * n)
        .map(|i| {
            f(
                TAU * (i % n) as f64 / n as f64,
                TAU * (i / n) as f64 / n as f64,
            )
        })
        .collect()
}
fn max_error(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max)
}
#[test]
fn derivative_of_sin_3x_is_3cos_3x() {
    let s = Spectral::new(32);
    assert!(
        max_error(
            &s.derivative(&grid(32, |x, _| (3. * x).sin()), 0),
            &grid(32, |x, _| 3. * (3. * x).cos())
        ) < 1e-12
    );
}
#[test]
fn poisson_of_two_cos_x_cos_y() {
    let s = Spectral::new(32);
    assert!(
        max_error(
            &s.poisson(&grid(32, |x, y| 2. * x.cos() * y.cos())),
            &grid(32, |x, y| x.cos() * y.cos())
        ) < 1e-12
    );
}
#[test]
fn fft_has_known_dc_coefficient_and_roundtrips() {
    let s = Spectral::new(32);
    let f = grid(32, |x, y| 1. + x.sin() + 0.3 * (2. * y).cos());
    let h = s.forward(&f);
    assert!((h[0].re - 1024.).abs() < 1e-10);
    assert!(max_error(&s.inverse(&h), &f) < 1e-12);
}
#[test]
fn nyquist_derivative_is_zero_on_even_real_grid() {
    let s = Spectral::new(32);
    assert!(s
        .derivative(&grid(32, |x, _| (16. * x).cos()), 0)
        .iter()
        .all(|v| v.abs() < 1e-12));
}
