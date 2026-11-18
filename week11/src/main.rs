mod gadgets;
mod magnet;
mod queens;

use clap::{Parser, ValueEnum};
use rand::{SeedableRng, rngs::StdRng};
use serde_json::json;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Clone, Debug, ValueEnum)]
enum Mode {
    Gadgets,
    Plan,
    Count,
    Sample,
}

#[derive(Clone, Debug, ValueEnum)]
enum Order {
    Greedy,
    Treesa,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, conflicts_with_all=["n", "l"])]
    input: Option<PathBuf>,
    #[arg(long, conflicts_with = "n")]
    l: Option<usize>,
    #[arg(long, requires = "l")]
    p_af: Option<f64>,
    #[arg(long)]
    n: Option<usize>,
    #[arg(long)]
    beta: Option<f64>,
    #[arg(long, value_enum, default_value = "treesa")]
    order: Order,
    #[arg(long)]
    samples: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let start = Instant::now();
    let mut rng = StdRng::seed_from_u64(args.seed);
    if matches!(args.mode, Mode::Sample) && args.samples.unwrap_or(0) == 0 {
        return Err("sample mode requires a positive --samples".into());
    }
    let (summary, evidence) = match args.mode {
        Mode::Gadgets => gadgets::run(),
        _ if let Some(n) = args.n => queens::run(n, &args, &mut rng)?,
        _ => magnet::run(&args, &mut rng)?,
    };
    let runtime = start.elapsed().as_millis();
    let mode = format!("{:?}", args.mode).to_lowercase();
    fs::create_dir_all(&args.out)?;
    let run = json!({
        "schema_version": 1, "mode": mode, "seed": args.seed, "runtime_ms": runtime,
        "parameters": {"input":args.input,"l":args.l,"p_af":args.p_af,"n":args.n,
            "beta":args.beta,"order":format!("{:?}",args.order).to_lowercase(),"samples":args.samples},
        "summary":summary, "evidence":evidence
    });
    fs::write(args.out.join("run.json"), serde_json::to_vec_pretty(&run)?)?;
    println!("mode\tmetric\tvalue\truntime_ms");
    for (metric, value) in summary.as_object().unwrap() {
        println!(
            "{mode}\t{metric}\t{}\t{runtime}",
            value
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| value.to_string())
        );
    }
    Ok(())
}
