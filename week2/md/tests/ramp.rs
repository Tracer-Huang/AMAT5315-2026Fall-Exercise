use md::fluid::{self, RunConfig};

#[test]
fn heating_schedule_has_exact_endpoints_and_linear_middle() {
    let c = RunConfig {
        temperature: 0.2,
        ramp_to: Some(1.2),
        steps: 20000,
        ..RunConfig::default()
    };
    assert_eq!(fluid::ramp_target(&c, 0).unwrap(), 0.2);
    assert_eq!(fluid::ramp_target(&c, 20000).unwrap(), 1.2);
    assert!((fluid::ramp_target(&c, 10000).unwrap() - 0.7).abs() < 1e-12);
    assert!((fluid::ramp_target(&c, 50).unwrap() - 0.2025).abs() < 1e-12);
    assert!(fluid::ramp_target(&c, 20001).is_err());
}

#[test]
fn ramp_changes_production_thermostat_and_is_not_an_nve_check() {
    let c = RunConfig {
        temperature: 0.2,
        ramp_to: Some(1.2),
        eq_steps: 50,
        steps: 100,
        sample_every: 50,
        ..RunConfig::default()
    };
    let frames = fluid::simulate_frames(&c).unwrap();
    for f in &frames {
        let thermo = 2.0 * f.e_kin / (2 * c.n - 2) as f64;
        let expected = if f.step == 50 { 0.7 } else { 1.2 };
        assert!(
            (thermo - expected).abs() < 1e-12,
            "step {}, temperature {} vs {}",
            f.step,
            thermo,
            expected
        );
    }
    let reason = fluid::analyze(&c, &frames).unwrap_err();
    assert!(reason.contains("heated"));
    assert_eq!(serde_json::to_value(&c).unwrap()["ramp_to"], 1.2);
}
