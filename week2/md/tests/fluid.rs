use md::{
    fluid::{self, RunConfig},
    periodic,
};

#[test]
fn triangular_lattice_has_the_specified_box_and_wrapped_positions() {
    let (p, b) = periodic::lattice(100, 0.8).unwrap();
    assert_eq!(p.len(), 100);
    assert!((100.0 / (b[0] * b[1]) - 0.8).abs() < 1e-12);
    assert!((b[0] - 12.01405707067377).abs() < 1e-10);
    assert!((b[1] - 10.404478625719542).abs() < 1e-10);
    assert!((p[10][0] - p[1][0] / 2.0).abs() < 1e-12);
    for x in p {
        for k in 0..2 {
            assert!(x[k] >= 0.0 && x[k] < b[k]);
        }
    }
    assert!(
        periodic::lattice(81, 0.8).is_err(),
        "odd periodic row count must fail"
    );
    assert!(periodic::lattice(99, 0.8).is_err());
}

#[test]
fn periodic_total_internal_force_vanishes() {
    let (mut p, b) = periodic::lattice(100, 0.8).unwrap();
    for (i, x) in p.iter_mut().enumerate() {
        x[0] = (x[0] + 0.02 * (i as f64).sin()).rem_euclid(b[0]);
        x[1] = (x[1] + 0.02 * (i as f64).cos()).rem_euclid(b[1]);
    }
    let mut f = vec![[0.0; 2]; p.len()];
    periodic::forces(&p, b, &mut f).unwrap();
    for axis in 0..2 {
        assert!(f.iter().map(|v| v[axis]).sum::<f64>().abs() < 1e-10);
    }
    assert!((periodic::minimum_image(9.0, 10.0) + 1.0).abs() < 1e-12);
}

#[test]
fn shifted_potential_is_continuous_just_inside_the_cutoff() {
    let mut f = [[0.0; 2]; 2];
    let inside = periodic::forces(&[[0.0, 0.0], [2.5 - 1e-7, 0.0]], [10.0, 10.0], &mut f).unwrap();
    assert!(
        inside.abs() < 1e-7,
        "inside limit must approach zero, got {inside}"
    );
    assert!(
        f[0][0].abs() > 0.01,
        "potential shift must not turn into force shifting"
    );
    let at = periodic::forces(&[[0.0, 0.0], [2.5, 0.0]], [10.0, 10.0], &mut f).unwrap();
    assert_eq!(at, 0.0);
    assert_eq!(f, [[0.0; 2]; 2]);
}

#[test]
fn production_saves_exact_steps_without_step_zero() {
    let c = RunConfig {
        eq_steps: 50,
        steps: 100,
        sample_every: 10,
        ..RunConfig::default()
    };
    let frames = fluid::simulate_frames(&c).unwrap();
    assert_eq!(frames.len(), 10);
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.step, (i + 1) * 10);
        assert!((f.t - f.step as f64 * c.dt).abs() < 1e-12);
        assert_eq!(f.pos.len(), c.n);
        assert_eq!(f.vel.len(), c.n);
    }
}

#[test]
fn default_contract_passes_recomputed_physics_and_rejects_tampering() {
    let c = RunConfig::default();
    let mut frames = fluid::simulate_frames(&c).unwrap();
    let report = fluid::analyze(&c, &frames).unwrap();
    println!("default contract development check: {report:?}");
    assert!(report.secular_drift < 2e-3);
    assert!((report.t_speed - 0.5).abs() < 0.05);
    assert!(report.chi2_per_dof < 2.0);
    assert!(report.passed);
    frames[0].e_pot += 1.0;
    assert!(
        fluid::analyze(&c, &frames).is_err(),
        "stored energy cannot be trusted"
    );
    frames[0].e_pot -= 1.0;
    frames[0].vel[0][0] += 1.0;
    assert!(
        fluid::analyze(&c, &frames).is_err(),
        "raw velocity edits must be detected"
    );
}
