use std::process::Command;
#[test]
fn exact_ad_agrees_with_enumeration() {
    let tmp = std::env::temp_dir().join(format!("week9-test-{}", std::process::id()));
    let out = Command::new(env!("CARGO_BIN_EXE_infer"))
        .args([
            "--model",
            "asia.uai",
            "--evidence",
            "AD",
            "--out",
            tmp.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let doc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tmp.join("run.json")).unwrap()).unwrap();
    assert!((doc["evidence_mass"].as_f64().unwrap() - 0.004501375).abs() < 1e-13);
    assert!(doc["max_marginal_error"].as_f64().unwrap() < 1e-10);
}
#[test]
fn bp_adx_converges_but_differs_from_exact() {
    let dir = std::env::temp_dir().join(format!("week9-bp-{}", std::process::id()));
    let out = Command::new(env!("CARGO_BIN_EXE_infer"))
        .args([
            "--model",
            "asia.uai",
            "--evidence",
            "ADX",
            "--method",
            "bp",
            "--damping",
            "0.5",
            "--tolerance",
            "1e-10",
            "--max-sweeps",
            "1000",
            "--out",
            dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("run.json")).unwrap()).unwrap();
    assert!(v["residual"].as_f64().unwrap() < 1e-10);
    assert!((v["max_marginal_error"].as_f64().unwrap() - 0.010611745).abs() < 1e-7);
}
#[test]
fn bp_supports_general_uai_and_evidence_file() {
    let dir = std::env::temp_dir().join(format!("w9tree-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("tree.uai"),
        "MARKOV\n2\n2 2\n2\n1 0\n2 0 1\n2\n0.2 0.8\n4\n0.9 0.1 0.3 0.7\n",
    )
    .unwrap();
    std::fs::write(dir.join("evid"), "0\n").unwrap();
    std::fs::write(dir.join("mar"), "MAR\n2 2 0.2 0.8 2 0.42 0.58\n").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_infer"))
        .args([
            "--model",
            dir.join("tree.uai").to_str().unwrap(),
            "--evidence-file",
            dir.join("evid").to_str().unwrap(),
            "--reference-mar",
            dir.join("mar").to_str().unwrap(),
            "--method",
            "bp",
            "--damping",
            "0.5",
            "--tolerance",
            "1e-10",
            "--max-sweeps",
            "1000",
            "--out",
            dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("run.json")).unwrap()).unwrap();
    assert!(v["max_marginal_error"].as_f64().unwrap() < 1e-9);
}
#[test]
fn pdf_ordering_commands_and_binding() {
    let dir = std::env::temp_dir().join(format!("w9pdf-{}", std::process::id()));
    let call = |args: Vec<String>| {
        Command::new(env!("CARGO_BIN_EXE_infer"))
            .args(args)
            .output()
            .unwrap()
    };
    let base = |d: &str| {
        vec![
            "--model".into(),
            "asia.uai".into(),
            "--evidence".into(),
            "AD".into(),
            "--out".into(),
            dir.join(d).to_str().unwrap().into(),
        ]
    };
    let mut a = base("random");
    a.extend([
        "--order".into(),
        "random".into(),
        "--seed".into(),
        "42".into(),
    ]);
    let x = call(a);
    assert!(x.status.success(), "{}", String::from_utf8_lossy(&x.stderr));
    let order = dir.join("random/order.json");
    assert!(order.exists());
    let c: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("random/complexity.json")).unwrap())
            .unwrap();
    assert!(c["steps"].as_array().unwrap().len() > 0);
    let mut a = base("analysis");
    a.extend(["--analyze".into(), order.to_str().unwrap().into()]);
    assert!(call(a).status.success());
    assert!(!dir.join("analysis/run.json").exists());
    let mut a = base("replay");
    a.extend(["--plan".into(), order.to_str().unwrap().into()]);
    assert!(call(a).status.success());
    let r: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("replay/run.json")).unwrap()).unwrap();
    assert!((r["contraction_mass"].as_f64().unwrap() - 0.004501375).abs() < 1e-12);
    let mut a = base("conflict");
    a.extend([
        "--order".into(),
        "greedy".into(),
        "--plan".into(),
        order.to_str().unwrap().into(),
    ]);
    assert!(!call(a).status.success());
    let mut a = base("mismatch");
    a[3] = "ADX".into();
    a.extend(["--plan".into(), order.to_str().unwrap().into()]);
    assert!(!call(a).status.success());
}
#[test]
fn checks_command_prints_tensor_results() {
    let out = Command::new(env!("CARGO_BIN_EXE_infer"))
        .arg("--checks")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(
        v["matrix_col_major"],
        serde_json::json!([4.0, 10.0, 4.0, 8.0])
    );
    assert!((v["ising"].as_f64().unwrap() - 12.60256209895193).abs() < 1e-12);
}
#[test]
fn rejects_inapplicable_bp_or_order_options() {
    let binary = env!("CARGO_BIN_EXE_infer");
    let run = |extra: &[&str]| {
        let mut c = Command::new(binary);
        c.args([
            "--model",
            "asia.uai",
            "--evidence",
            "AD",
            "--out",
            std::env::temp_dir().to_str().unwrap(),
        ]);
        c.args(extra).output().unwrap().status.success()
    };
    assert!(!run(&["--damping", "0.5"]));
    assert!(!run(&["--order", "greedy", "--seed", "42"]));
    assert!(!run(&["--order", "random"]));
    assert!(!run(&[
        "--method",
        "bp",
        "--order",
        "greedy",
        "--damping",
        "0.5",
        "--tolerance",
        "1e-10",
        "--max-sweeps",
        "5"
    ]));
}
