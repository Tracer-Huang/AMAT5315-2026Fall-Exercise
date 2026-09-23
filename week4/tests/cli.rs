use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
fn field(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_field"))
        .args(args)
        .output()
        .unwrap()
}
fn run(input: &[u8], args: &[&str]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_fluid"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let data = input.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&data);
    });
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap();
    output
}
fn directory(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("week4-test-{}-{}", std::process::id(), name))
}
#[test]
fn pipeline_preserves_metadata_times_and_tracer_schema() {
    let f = field(&[
        "random", "--n", "16", "--seed", "2026", "--k-min", "2", "--k-max", "4",
    ]);
    assert!(f.status.success(), "{}", String::from_utf8_lossy(&f.stderr));
    let out = directory("pipeline");
    let outstr = out.to_str().unwrap();
    let r = run(
        &f.stdout,
        &[
            "--nu",
            "0.004",
            "--dt",
            "0.007",
            "--t-end",
            "0.1",
            "--every",
            "0.03",
            "--out",
            outstr,
            "--tracers",
            "20",
            "--tracer-seed",
            "7",
            "--tracer-every",
            "2",
        ],
    );
    assert!(r.status.success(), "{}", String::from_utf8_lossy(&r.stderr));
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
    assert_eq!(meta["seed"], 2026);
    assert_eq!(meta["k_band"], serde_json::json!([2, 4]));
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(out.join("fields.jsonl"))
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(
        rows.iter()
            .map(|r| r["t"].as_f64().unwrap())
            .collect::<Vec<_>>(),
        vec![0., 0.03, 0.06, 0.09, 0.1]
    );
    assert!(rows.iter().all(|r| r["u"].as_array().unwrap().len() == 256));
    let tracers = std::fs::read_to_string(out.join("tracers.jsonl")).unwrap();
    assert_eq!(tracers.lines().count(), 3);
    for l in tracers.lines() {
        let r: serde_json::Value = serde_json::from_str(l).unwrap();
        for d in ["x", "y"] {
            assert!(r[d]
                .as_array()
                .unwrap()
                .iter()
                .all(|x| (0. ..std::f64::consts::TAU).contains(&x.as_f64().unwrap())));
        }
    }
    let again = run(
        &f.stdout,
        &[
            "--nu", "0.004", "--dt", "0.01", "--t-end", "0.1", "--every", "0.1", "--out", outstr,
        ],
    );
    assert!(!again.status.success());
    assert!(String::from_utf8_lossy(&again.stderr).contains("not empty"));
    std::fs::remove_dir_all(out).unwrap();
}
#[test]
fn invalid_flags_and_missing_explicit_values_are_rejected() {
    for args in [
        vec!["taylor-green"],
        vec!["taylor-green", "--n", "12"],
        vec![
            "random", "--n", "16", "--seed", "2", "--k-min", "2", "--k-max", "6",
        ],
        vec!["taylor-green", "--n", "16", "--typo", "1"],
    ] {
        assert!(!field(&args).status.success());
    }
    let f = field(&["taylor-green", "--n", "16"]);
    assert!(f.status.success());
    for dt in ["0", "-0.1", "NaN"] {
        let out = directory("invalid");
        let r = run(
            &f.stdout,
            &[
                "--nu",
                "0.1",
                "--dt",
                dt,
                "--t-end",
                "1",
                "--every",
                "0.1",
                "--out",
                out.to_str().unwrap(),
            ],
        );
        assert!(!r.status.success());
        assert!(!out.exists());
    }
    assert!(!run(&f.stdout, &["--nu", "0.1"]).status.success());
}

#[test]
fn part5_adds_tracers_to_same_run_without_replacing_fields() {
    let f = field(&[
        "random", "--n", "16", "--seed", "2026", "--k-min", "2", "--k-max", "4",
    ]);
    assert!(f.status.success());
    let out = directory("append-tracers");
    let args = [
        "--nu",
        "0.004",
        "--dt",
        "0.01",
        "--t-end",
        "0.1",
        "--every",
        "0.1",
        "--out",
        out.to_str().unwrap(),
    ];
    assert!(run(&f.stdout, &args).status.success());
    let before = std::fs::read(out.join("fields.jsonl")).unwrap();
    let before_meta = std::fs::read(out.join("run.json")).unwrap();
    let mut added = args.to_vec();
    added.extend([
        "--tracers",
        "20",
        "--tracer-seed",
        "7",
        "--tracer-every",
        "1",
    ]);
    let result = run(&f.stdout, &added);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(before, std::fs::read(out.join("fields.jsonl")).unwrap());
    assert_eq!(
        before_meta,
        std::fs::read(out.join("run.before-tracers.json")).unwrap()
    );
    let positions = std::fs::read(out.join("tracers.jsonl")).unwrap();
    assert!(run(&f.stdout, &added).status.success());
    assert_eq!(positions, std::fs::read(out.join("tracers.jsonl")).unwrap());
    let different = field(&[
        "random", "--n", "16", "--seed", "2027", "--k-min", "2", "--k-max", "4",
    ]);
    assert!(!run(&different.stdout, &added).status.success());
    assert_eq!(before, std::fs::read(out.join("fields.jsonl")).unwrap());
    let mut changed_input: serde_json::Value = serde_json::from_slice(&f.stdout).unwrap();
    changed_input["u"][0] = serde_json::json!(changed_input["u"][0].as_f64().unwrap() + 0.1);
    let rejected = run(&serde_json::to_vec(&changed_input).unwrap(), &added);
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("fields differ"));
    assert_eq!(before, std::fs::read(out.join("fields.jsonl")).unwrap());
    std::fs::remove_dir_all(out).unwrap();
}
