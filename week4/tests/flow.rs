use std::f64::consts::TAU;
use week4_fluid::{flow::*, Spectral};
#[test]
fn taylor_green_decays_to_exact_field_energy_and_divergence() {
    let n = 64;
    let s = Spectral::new(n);
    let (u0, v0) = taylor_green(n);
    assert!((energy(&u0, &v0) - 0.25).abs() < 1e-14);
    let mut w = vorticity(&s, &u0, &v0);
    for _ in 0..100 {
        step(&s, &mut w, &mut [], 0.1, 0.01);
    }
    let (u, v) = s.velocity(&w);
    let exact = (-0.2_f64).exp();
    assert!((energy(&u, &v) - 0.25 * (-0.4_f64).exp()).abs() < 1e-6);
    assert!(u
        .iter()
        .zip(&u0)
        .chain(v.iter().zip(&v0))
        .all(|(a, b)| (a - b * exact).abs() < 1e-10));
    let ux = s.derivative(&u, 0);
    let vy = s.derivative(&v, 1);
    assert!(ux.iter().zip(vy).all(|(a, b)| (a + b).abs() < 1e-10));
}
#[test]
fn seeded_field_has_same_physical_modes_on_different_grids() {
    let (a, b) = random_field(32, 2026, 2, 6);
    let (c, d) = random_field(64, 2026, 2, 6);
    assert!((energy(&a, &b) - 0.5).abs() < 1e-12);
    for y in 0..32 {
        for x in 0..32 {
            assert!((a[y * 32 + x] - c[2 * y * 64 + 2 * x]).abs() < 1e-12);
            assert!((b[y * 32 + x] - d[2 * y * 64 + 2 * x]).abs() < 1e-12);
        }
    }
    assert_ne!(a, random_field(32, 2027, 2, 6).0);
    let s = Spectral::new(32);
    let w = vorticity(&s, &a, &b);
    assert!(w
        .iter()
        .enumerate()
        .all(|(i, z)| (4. ..=36.).contains(&s.k2(i)) || z.norm() < 1e-10));
}
#[test]
fn random_dynamics_are_not_pure_diffusion() {
    let s = Spectral::new(32);
    let (u, v) = random_field(32, 2026, 2, 6);
    let w0 = vorticity(&s, &u, &v);
    let mut w = w0.clone();
    for _ in 0..100 {
        step(&s, &mut w, &mut [], 0.004, 0.01);
    }
    let diff = w
        .iter()
        .zip(&w0)
        .enumerate()
        .map(|(i, (a, b))| (*a - *b * (-0.004 * s.k2(i)).exp()).norm_sqr())
        .sum::<f64>();
    let norm = w0.iter().map(|z| z.norm_sqr()).sum::<f64>();
    assert!((diff / norm).sqrt() > 0.2);
}
#[test]
fn interpolation_is_periodic_and_reproduces_constant_velocity() {
    let s = Spectral::new(16);
    let u = vec![2.; 256];
    let v = vec![-3.; 256];
    assert_eq!(interpolate(&s, &u, &v, [TAU + 0.2, -0.4]), [2., -3.]);
}
#[test]
fn tracers_follow_steady_shear_with_periodic_wrap() {
    let s = Spectral::new(32);
    let u: Vec<_> = (0..1024)
        .map(|i| (TAU * (i / 32) as f64 / 32.).sin())
        .collect();
    let mut w = vorticity(&s, &u, &vec![0.; 1024]);
    let mut p = [[TAU - 0.1, TAU / 4.]];
    for _ in 0..20 {
        step(&s, &mut w, &mut p, 0., 0.01);
    }
    assert!((p[0][0] - 0.1).abs() < 1e-12);
    assert!((p[0][1] - TAU / 4.).abs() < 1e-12);
}

#[test]
fn tracer_uses_time_dependent_rk_stage_velocities() {
    let s = Spectral::new(32);
    let u: Vec<_> = (0..1024)
        .map(|i| (TAU * (i / 32) as f64 / 32.).sin())
        .collect();
    let mut w = vorticity(&s, &u, &vec![0.; 1024]);
    let mut p = [[1., TAU / 4.]];
    // nu*dt*max(k^2)=2.0 is inside RK4's negative-real stability interval.
    for _ in 0..50 {
        step(&s, &mut w, &mut p, 0.5, 0.02);
    }
    let exact = 1. + (1. - (-0.5_f64).exp()) / 0.5;
    assert!(
        (p[0][0] - exact).abs() < 2e-10,
        "measured={} exact={exact}",
        p[0][0]
    );
}
