use seismic::{Action, schedule, step};
#[test]
fn update_matches_centered_damped_equation_and_zero_boundary() {
    let (nx, nz) = (5, 5);
    let mut cur = vec![0.; 25];
    cur[12] = 1.;
    let prev = vec![0.2; 25];
    let c = vec![1.8; 25];
    let sigma = vec![0.4; 25];
    let q = vec![0.3; 25];
    let out = step(&prev, &cur, &c, &sigma, &q, nx, nz, 1., 0.2);
    let expected =
        (2. - (1. - 0.4 * 0.2) * 0.2 + 0.04 * (1.8 * 1.8 * (-4.) + 0.3)) / (1. + 0.4 * 0.2);
    assert!((out[12] - expected).abs() < 1e-14);
    assert_eq!(out[0], 0.);
    assert_eq!(out[24], 0.);
}
#[test]
fn treeverse_independent_schedule_invariants() {
    use std::collections::BTreeSet;
    for n in 1..45 {
        for budget in 1..7 {
            let actions = schedule(n, budget);
            let mut saved = BTreeSet::from([0]);
            let mut work = 0;
            let mut grads = vec![];
            for a in actions {
                match a {
                    Action::Restore(s) => {
                        assert!(saved.contains(&s));
                        work = s
                    }
                    Action::Call(s) => {
                        assert_eq!(work, s);
                        work += 1;
                    }
                    Action::Store(s) => {
                        assert_eq!(work, s);
                        assert!(saved.insert(s));
                        assert!(saved.len() <= budget + 1);
                    }
                    Action::Grad(s) => {
                        assert!(saved.contains(&s));
                        grads.push(s);
                    }
                    Action::Fetch(s) => {
                        assert_ne!(s, 0);
                        assert!(saved.remove(&s));
                    }
                }
            }
            assert_eq!(grads, (0..n).rev().collect::<Vec<_>>());
            assert_eq!(saved, BTreeSet::from([0]));
        }
    }
}
#[test]
fn prescribed_treeverse_work_counts() {
    for (b, w) in [(1, 28680), (3, 1695), (5, 990), (10, 642)] {
        assert_eq!(
            schedule(240, b)
                .iter()
                .filter(|x| matches!(x, Action::Call(_)))
                .count(),
            w
        );
    }
}
#[test]
fn real_enzyme_cube_forward_and_reverse() {
    assert_eq!(seismic::cube_probe(2.), [8., 12., 12.]);
}
#[test]
fn enzyme_timestep_jvp_finite_difference_and_vjp_transpose() {
    let m = 30;
    let model = seismic::Model {
        nx: 6,
        nz: 5,
        dx: 1.,
        dt: 0.1,
        c: (0..m).map(|i| 1.4 + i as f64 / 100.).collect(),
        sigma: vec![0.2; m],
    };
    let s = (0..2 * m)
        .map(|i| (i as f64 * 0.23).sin())
        .collect::<Vec<_>>();
    let ds = (0..2 * m)
        .map(|i| (i as f64 * 0.17).cos())
        .collect::<Vec<_>>();
    let dc = (0..m).map(|i| (i as f64 * 0.19).sin()).collect::<Vec<_>>();
    let q = vec![0.3; m];
    let w = (0..2 * m)
        .map(|i| (i as f64 * 0.31).sin())
        .collect::<Vec<_>>();
    let mut out = vec![0.; 2 * m];
    let mut dout = vec![0.; 2 * m];
    model.jvp(&s, &ds, &dc, &q, &mut out, &mut dout);
    let h = 1e-5;
    let mut plus = model.clone();
    let mut minus = model.clone();
    for i in 0..m {
        plus.c[i] += h * dc[i];
        minus.c[i] -= h * dc[i];
    }
    let sp = s
        .iter()
        .zip(&ds)
        .map(|(a, b)| a + h * b)
        .collect::<Vec<_>>();
    let sm = s
        .iter()
        .zip(&ds)
        .map(|(a, b)| a - h * b)
        .collect::<Vec<_>>();
    let mut yp = vec![0.; 2 * m];
    let mut ym = yp.clone();
    plus.forward(&sp, &q, &mut yp);
    minus.forward(&sm, &q, &mut ym);
    for i in 0..2 * m {
        assert!(((yp[i] - ym[i]) / (2. * h) - dout[i]).abs() < 1e-9);
    }
    let (as_, ac) = model.vjp(&s, &q, &w);
    let left = dout.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>();
    let right = as_
        .iter()
        .zip(&ds)
        .chain(ac.iter().zip(&dc))
        .map(|(a, b)| a * b)
        .sum::<f64>();
    assert!((left - right).abs() < 1e-12);
}
