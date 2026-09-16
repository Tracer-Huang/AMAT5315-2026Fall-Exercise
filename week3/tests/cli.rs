use std::{fs, process::Command};
#[test]
fn contract_seed_and_frames() {
    let root = std::env::temp_dir().join(format!("ising-test-{}", std::process::id()));
    for (n, seed, update) in [
        ("a", "2026", "metropolis"),
        ("b", "2026", "metropolis"),
        ("c", "2027", "metropolis"),
        ("w", "2026", "wolff"),
    ] {
        let out = root.join(n);
        let s = Command::new(env!("CARGO_BIN_EXE_ising"))
            .args([
                "--update",
                update,
                "--l",
                "4",
                "--t-from",
                "2",
                "--t-to",
                "2.1",
                "--t-step",
                "0.1",
                "--discard",
                "3",
                "--measure",
                "10",
                "--every",
                "5",
                "--seed",
                seed,
                "--out",
            ])
            .arg(&out)
            .output()
            .unwrap();
        assert!(s.status.success(), "{}", String::from_utf8_lossy(&s.stderr));
        let run: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
        assert_eq!(run["sample_every"], 1);
        let rows = fs::read_to_string(out.join("series.jsonl")).unwrap();
        assert_eq!(rows.lines().count(), 20);
        let frames = fs::read_to_string(out.join("spins.jsonl")).unwrap();
        assert_eq!(frames.lines().count(), 4);
        let v: serde_json::Value = serde_json::from_str(frames.lines().next().unwrap()).unwrap();
        assert_eq!(v["sweep"], 8);
        assert_eq!(v["spins"].as_array().unwrap().len(), 16);
    }
    assert_eq!(
        fs::read(root.join("a/series.jsonl")).unwrap(),
        fs::read(root.join("b/series.jsonl")).unwrap()
    );
    assert_ne!(
        fs::read(root.join("a/series.jsonl")).unwrap(),
        fs::read(root.join("c/series.jsonl")).unwrap()
    );
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn rejects_invalid_or_missing_flags() {
    assert!(!Command::new(env!("CARGO_BIN_EXE_ising"))
        .arg("--bad")
        .output()
        .unwrap()
        .status
        .success());
}

#[test]
fn upper_bound_only_when_reached_and_default_no_frames() {
    let root = std::env::temp_dir().join(format!("ising-grid-{}", std::process::id()));
    let output = Command::new(env!("CARGO_BIN_EXE_ising"))
        .args([
            "--update",
            "metropolis",
            "--l",
            "2",
            "--t-from",
            "2.0",
            "--t-to",
            "2.25",
            "--t-step",
            "0.1",
            "--discard",
            "0",
            "--measure",
            "2",
            "--seed",
            "1",
            "--out",
        ])
        .arg(&root)
        .output()
        .unwrap();
    assert!(output.status.success());
    let meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("run.json")).unwrap()).unwrap();
    assert_eq!(meta["t_grid"], serde_json::json!([2.0, 2.1, 2.2]));
    assert!(!root.join("spins.jsonl").exists());
    fs::remove_dir_all(root).unwrap();
}
