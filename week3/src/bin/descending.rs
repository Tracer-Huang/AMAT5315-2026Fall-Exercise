//! Non-neural extension: random-start cooling, separate from the fixed ising contract.
use ising::{acceptance, Lattice};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("descending L DISCARD MEASURE SEED OUT; T=3.5 to 1.5 by -0.05".into());
    }
    let l: usize = args[0].parse()?;
    let discard: usize = args[1].parse()?;
    let measure: usize = args[2].parse()?;
    let seed: u64 = args[3].parse()?;
    let out = PathBuf::from(&args[4]);
    if !(2..=4096).contains(&l) || measure == 0 {
        return Err("invalid L or measure".into());
    }
    let total = discard.checked_add(measure).ok_or("step count overflow")?;
    fs::create_dir_all(&out)?;
    let start = Instant::now();
    let n = (l * l) as f64;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = Lattice::from_spins(
        l,
        (0..l * l)
            .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
            .collect(),
    );
    let grid: Vec<f64> = (0..41).map(|i| (350 - 5 * i) as f64 / 100.).collect();
    fs::write(
        out.join("run.json"),
        serde_json::to_string_pretty(
            &json!({"L":l,"update":"metropolis","initial":"random","t_grid":grid,"discard":discard,"measure":measure,"seed":seed,"sample_every":1,"time_unit":"sweep","extension":"descending"}),
        )?,
    )?;
    let mut series = BufWriter::new(File::create(out.join("series.jsonl"))?);
    let mut frames = BufWriter::new(File::create(out.join("spins.jsonl"))?);
    let mut cumulative = 0;
    for t in grid {
        let table = acceptance(t);
        for step in 1..=total {
            state.metropolis(&mut rng, &table);
            cumulative += 1;
            if step > discard {
                let index = step - discard;
                let m = state.magnet as f64 / n;
                let e = state.energy as f64 / n;
                writeln!(
                    series,
                    "{{\"L\":{l},\"T\":{t},\"sweep\":{index},\"M\":{m:.6},\"E\":{e:.6}}}"
                )?;
                if index.is_multiple_of((measure / 10).max(1)) {
                    writeln!(
                        frames,
                        "{}",
                        json!({"L":l,"T":t,"sweep":cumulative,"m":m,"spins":state.spins})
                    )?;
                }
            }
        }
    }
    series.flush()?;
    frames.flush()?;
    fs::write(
        out.join("timing.json"),
        json!({"wall_seconds":start.elapsed().as_secs_f64(),"args":args}).to_string(),
    )?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("descending: {e}");
        std::process::exit(2);
    }
}
