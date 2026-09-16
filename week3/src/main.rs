use ising::{acceptance, Lattice};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw == ["--help"] {
        println!("ising --update metropolis|wolff --l N --t-from T --t-to T --t-step D --discard N --measure N --seed N --out DIR [--every N]");
        return Ok(());
    }
    let allowed = [
        "update", "l", "t-from", "t-to", "t-step", "discard", "measure", "seed", "out", "every",
    ];
    if !raw.len().is_multiple_of(2) {
        return Err("flags require values".into());
    }
    let mut a = HashMap::new();
    for pair in raw.chunks(2) {
        let k = pair[0].strip_prefix("--").ok_or("expected flag")?;
        if !allowed.contains(&k) || a.insert(k, pair[1].as_str()).is_some() {
            return Err("unknown or duplicate flag".into());
        }
    }
    let get = |k| a.get(k).copied().ok_or_else(|| format!("missing --{k}"));
    let update = get("update")?;
    if !["metropolis", "wolff"].contains(&update) {
        return Err("invalid update".into());
    }
    let l: usize = get("l")?.parse()?;
    if !(2..=4096).contains(&l) {
        return Err("l must be 2..4096".into());
    }
    let from: f64 = get("t-from")?.parse()?;
    let to: f64 = get("t-to")?.parse()?;
    let dt: f64 = get("t-step")?.parse()?;
    if ![from, to, dt].iter().all(|x| x.is_finite() && *x > 0.) || to < from {
        return Err("finite positive ascending temperatures required".into());
    }
    let count = ((to - from) / dt + 1e-9).floor() as usize + 1;
    if count > 100000 {
        return Err("temperature grid too large".into());
    }
    let grid: Vec<f64> = (0..count)
        .map(|i| ((from + dt * i as f64) * 1e12).round() / 1e12)
        .collect();
    let discard: usize = get("discard")?.parse()?;
    let measure: usize = get("measure")?.parse()?;
    if measure == 0 {
        return Err("measure must be positive".into());
    }
    let every: usize = a.get("every").unwrap_or(&"0").parse()?;
    let seed: u64 = get("seed")?.parse()?;
    let out = PathBuf::from(get("out")?);
    fs::create_dir_all(&out)?;
    let meta = json!({"L":l,"update":update,"t_grid":grid,"discard":discard,"measure":measure,"seed":seed,"sample_every":1,"time_unit":if update=="metropolis" {"sweep"}else{"cluster_flip"}});
    fs::write(out.join("run.json"), serde_json::to_string_pretty(&meta)?)?;
    let mut series = BufWriter::new(File::create(out.join("series.jsonl"))?);
    let mut frames = if every > 0 {
        Some(BufWriter::new(File::create(out.join("spins.jsonl"))?))
    } else {
        None
    };
    let mut state = Lattice::new(l);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut cumulative = 0usize;
    let n = (l * l) as f64;
    println!(
        "T\tmean_abs_M\t{}",
        if update == "metropolis" {
            "acceptance"
        } else {
            "mean_cluster_size"
        }
    );
    for t in grid {
        let table = acceptance(t);
        let p = 1. - (-2. / t).exp();
        let mut acc = 0usize;
        let mut cluster_sum = 0usize;
        let mut mean = 0.;
        for step in 1..=discard + measure {
            let c = if update == "metropolis" {
                state.metropolis(&mut rng, &table)
            } else {
                state.wolff(&mut rng, p)
            };
            acc += c;
            cumulative += 1;
            if step > discard {
                let index = step - discard;
                let m = state.magnet as f64 / n;
                let e = state.energy as f64 / n;
                mean += m.abs();
                cluster_sum += c;
                if update == "metropolis" {
                    writeln!(
                        series,
                        "{{\"L\":{l},\"T\":{t},\"sweep\":{index},\"M\":{m:.6},\"E\":{e:.6}}}"
                    )?;
                } else {
                    writeln!(series,"{{\"L\":{l},\"T\":{t},\"sweep\":{index},\"M\":{m:.6},\"E\":{e:.6},\"cluster_size\":{c}}}")?;
                }
                if every > 0 && index.is_multiple_of(every) {
                    let row = json!({"L":l,"T":t,"sweep":cumulative,"m":m,"spins":state.spins});
                    let f = frames.as_mut().unwrap();
                    serde_json::to_writer(&mut *f, &row)?;
                    writeln!(f)?;
                }
            }
        }
        let rate = if update == "metropolis" {
            acc as f64 / ((discard + measure) as f64 * n)
        } else {
            cluster_sum as f64 / measure as f64
        };
        println!("{t:.6}\t{:.6}\t{rate:.6}", mean / measure as f64);
    }
    series.flush()?;
    if let Some(mut f) = frames {
        f.flush()?;
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("ising: {e}");
        std::process::exit(2);
    }
}
