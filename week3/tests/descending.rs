use std::{fs, process::Command};
#[test]
fn random_start_descending_recording_and_repeatability() {
    let root = std::env::temp_dir().join(format!("ising-descending-{}", std::process::id()));
    for name in ["a", "b"] {
        let out = root.join(name);
        let result = Command::new(env!("CARGO_BIN_EXE_descending"))
            .args(["4", "5", "20", "2026"])
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let meta: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(out.join("run.json")).unwrap()).unwrap();
        assert_eq!(meta["initial"], "random");
        let ts = meta["t_grid"].as_array().unwrap();
        assert_eq!(ts.len(), 41);
        assert_eq!(ts[0], 3.5);
        assert_eq!(ts[40], 1.5);
        assert_eq!(
            fs::read_to_string(out.join("series.jsonl"))
                .unwrap()
                .lines()
                .count(),
            820
        );
        assert_eq!(
            fs::read_to_string(out.join("spins.jsonl"))
                .unwrap()
                .lines()
                .count(),
            410
        );
    }
    assert_eq!(
        fs::read(root.join("a/series.jsonl")).unwrap(),
        fs::read(root.join("b/series.jsonl")).unwrap()
    );
    fs::remove_dir_all(root).unwrap();
}
