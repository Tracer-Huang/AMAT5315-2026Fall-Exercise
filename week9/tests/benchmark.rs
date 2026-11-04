use std::process::Command;
#[test]
fn greedy_plan_asia_matches_known_evidence_mass() {
    let dir = std::env::temp_dir().join(format!("w9bench-{}", std::process::id()));
    let o = Command::new(env!("CARGO_BIN_EXE_benchmark"))
        .args(["asia.uai", "0=0,7=0", "greedy", dir.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("result.json")).unwrap()).unwrap();
    assert!((v["log10_z"].as_f64().unwrap() - 0.004501375_f64.log10()).abs() < 1e-10);
}
#[test]
fn random_and_replayed_plans_agree() {
    let base = std::env::temp_dir().join(format!("w9random-{}", std::process::id()));
    let run = |method: &str, dir: &str| {
        let o = Command::new(env!("CARGO_BIN_EXE_benchmark"))
            .args(["asia.uai", "0=0,7=0", method, dir])
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(format!("{dir}/result.json")).unwrap(),
        )
        .unwrap()
    };
    let a = base.join("random");
    let b = base.join("replay");
    let x = run("random:42", a.to_str().unwrap());
    let method = format!("replay:{}", a.join("plan-z.json").display());
    let y = run(&method, b.to_str().unwrap());
    assert!((x["log10_z"].as_f64().unwrap() - y["log10_z"].as_f64().unwrap()).abs() < 1e-12);
}
