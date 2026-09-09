use md::dynamics::{ForwardEuler, Integrator, System, VelocityVerlet, advance};

fn experiment(method: &impl Integrator, steps: usize) -> (f64, f64) {
    let mut system = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0; 2]; 2]).unwrap();
    let e0 = system.total_energy();
    let mut maximum = 0.0_f64;
    let mut final_error = 0.0;
    for _ in 0..steps {
        advance(method, &mut system, 0.01).unwrap();
        final_error = (system.total_energy() - e0) / e0.abs();
        maximum = maximum.max(final_error.abs());
    }
    (maximum, final_error)
}

#[test]
fn dimer_same_trait_same_state_distinguishes_euler_and_verlet() {
    let euler = experiment(&ForwardEuler, 500);
    let verlet = experiment(&VelocityVerlet, 500);
    println!(
        "dimer dt=0.01, 500 steps: Euler final={}, Verlet max={}",
        euler.1, verlet.0
    );
    assert!(euler.1 > 0.5, "Euler final relative error: {}", euler.1);
    assert!(
        verlet.0 < 1e-3,
        "Verlet max absolute relative error: {}",
        verlet.0
    );
}

#[test]
fn verlet_energy_remains_bounded_for_ten_times_longer() {
    let (maximum, _) = experiment(&VelocityVerlet, 5000);
    println!("Verlet 5000-step max absolute relative error={maximum}");
    assert!(maximum < 1e-3);
}

#[test]
fn free_particle_follows_constant_velocity_through_the_trait() {
    fn check(method: &impl Integrator) {
        let mut system = System::new(vec![[0.0, 0.0]], vec![[1.0, -2.0]]).unwrap();
        advance(method, &mut system, 0.5).unwrap();
        assert_eq!(system.positions, vec![[0.5, -1.0]]);
        assert_eq!(system.velocities, vec![[1.0, -2.0]]);
    }
    check(&ForwardEuler);
    check(&VelocityVerlet);
}
