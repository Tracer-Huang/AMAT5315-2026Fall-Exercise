use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn cli_writes_readable_frames_and_rejects_malformed_input() {
    let dir = std::env::temp_dir().join(format!(
        "amat5315-cli-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let status = Command::new(env!("CARGO_BIN_EXE_md"))
        .args([
            "run",
            "--eq-steps",
            "50",
            "--steps",
            "100",
            "--sample-every",
            "10",
            "--out",
        ])
        .arg(&dir)
        .status()
        .unwrap();
    assert!(status.success());
    let run: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(dir.join("run.json")).expect("binary must write run.json"),
    )
    .unwrap();
    assert_eq!(run["integrator"], "velocity-verlet");
    assert_eq!(run["seed"], 2026);
    let (config, frames) = md::fluid::read_run(&dir).unwrap();
    assert_eq!(config.n, 100);
    assert_eq!(frames.len(), 10);
    assert_eq!(frames[0].step, 10);
    // Deliberately corrupt only this test's unique temporary fixture.
    fs::write(dir.join("traj.jsonl"), "this is not JSON\n").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_md"))
        .arg("check")
        .arg(&dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("error"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_rejects_unknown_flags_and_invalid_numbers() {
    for args in [
        vec!["run", "--not-a-flag", "2"],
        vec!["run", "--dt", "NaN"],
        vec!["run", "--sample-every", "0"],
    ] {
        let result = Command::new(env!("CARGO_BIN_EXE_md"))
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success());
    }
}

#[test]
fn cli_rejects_excessive_particle_count_before_allocating() {
    let result = Command::new(env!("CARGO_BIN_EXE_md"))
        .args(["run", "--n", "1000000000000000000"])
        .output()
        .unwrap();
    assert_eq!(
        result.status.code(),
        Some(1),
        "invalid input must return an error, not panic: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("error:") && stderr.contains("n is too large"));
}

#[test]
fn cli_defaults_to_cells_and_keeps_the_naive_flag() {
    let root = std::env::temp_dir().join(format!(
        "amat5315-force-cli-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    for method in ["cells", "naive"] {
        let dir = root.join(method);
        let mut command = Command::new(env!("CARGO_BIN_EXE_md"));
        command
            .args([
                "run",
                "--eq-steps",
                "20",
                "--steps",
                "20",
                "--sample-every",
                "10",
                "--out",
            ])
            .arg(&dir);
        if method == "naive" {
            command.args(["--force", "naive"]);
        }
        let status = command.status().unwrap();
        assert!(status.success());
        let metadata: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
        assert_eq!(metadata["force"], method);
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn cli_records_the_requested_heating_target() {
    let dir = std::env::temp_dir().join(format!(
        "amat5315-ramp-cli-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let result = Command::new(env!("CARGO_BIN_EXE_md"))
        .args([
            "run",
            "--temperature",
            "0.2",
            "--ramp-to",
            "1.2",
            "--eq-steps",
            "50",
            "--steps",
            "100",
            "--out",
        ])
        .arg(&dir)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let (config, frames) = md::fluid::read_run(&dir).unwrap();
    assert_eq!(config.ramp_to, Some(1.2));
    let thermo = 2.0 * frames.last().unwrap().e_kin / (2 * config.n - 2) as f64;
    assert!((thermo - 1.2).abs() < 1e-12);
    fs::remove_dir_all(dir).unwrap();
}
