use md::{
    dynamics::MdResult,
    fluid::{self, RunConfig},
    periodic,
};
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

const HELP: &str = "md: AMAT5315 two-dimensional molecular dynamics

md run [--n 100] [--rho 0.8] [--temperature 0.5] [--dt 0.01]
       [--eq-steps 2000] [--steps 10000] [--sample-every 50]
       [--seed 2026] [--force cells|naive] [--out artifacts]
md check <trajectory-directory>
md video <trajectory-directory> --out <video.mp4>

The unheated default contract uses velocity-Verlet and seed 2026.
check recomputes raw-frame physics and applies the PDF's three T=0.5 bounds.
Repeated runs preserve previous run.json/traj.jsonl in an output history folder.
Video rendering uses the course .venv, MD_PYTHON, or python3 and an ffmpeg encoder.";

fn parse<T: std::str::FromStr>(value: &str, name: &str) -> MdResult<T> {
    value
        .parse()
        .map_err(|_| format!("invalid value for {name}: {value}"))
}

fn execute() -> MdResult<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        println!("{}", md::greeting());
        return Ok(());
    }
    if args.iter().any(|x| x == "--help" || x == "-h") {
        println!("{HELP}");
        return Ok(());
    }
    if args == ["--version"] {
        println!("md {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    match args[0].as_str() {
        "run" => {
            let mut config = RunConfig::default();
            let mut out = PathBuf::from("artifacts");
            let mut index = 1;
            while index < args.len() {
                let name = &args[index];
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| format!("missing value for {name}"))?;
                match name.as_str() {
                    "--n" => config.n = parse(value, name)?,
                    "--rho" => config.rho = parse(value, name)?,
                    "--temperature" => config.temperature = parse(value, name)?,
                    "--dt" => config.dt = parse(value, name)?,
                    "--eq-steps" => config.eq_steps = parse(value, name)?,
                    "--steps" => config.steps = parse(value, name)?,
                    "--sample-every" => config.sample_every = parse(value, name)?,
                    "--seed" => config.seed = parse(value, name)?,
                    "--force" => config.force = value.parse()?,
                    "--out" => out = PathBuf::from(value),
                    _ => return Err(format!("unknown flag {name}; use md --help")),
                }
                index += 2;
            }
            config.box_size = periodic::lattice(config.n, config.rho)?.1;
            config.validate()?;
            let count = fluid::write_run(&config, &out)?;
            println!(
                "wrote {}/run.json + traj.jsonl ({count} production frames, velocity-verlet, {})",
                out.display(),
                config.force.as_str()
            );
        }
        "check" => {
            if args.len() != 2 {
                return Err("usage: md check <trajectory-directory>".into());
            }
            let (config, frames) = fluid::read_run(Path::new(&args[1]))?;
            let report = fluid::analyze(&config, &frames)?;
            println!(
                "secular drift = {:.8e}; limit < 2.0e-3",
                report.secular_drift
            );
            println!(
                "T_speed = {:.8}; |T_speed - 0.5| = {:.8}; limit < 0.05",
                report.t_speed,
                (report.t_speed - 0.5).abs()
            );
            println!("chi2 / 22 = {:.8}; limit < 2.0", report.chi2_per_dof);
            println!("{}", if report.passed { "PASS" } else { "FAIL" });
            if !report.passed {
                return Err("physics acceptance failed".into());
            }
        }
        "video" => {
            if args.len() != 4 || args[2] != "--out" {
                return Err("usage: md video <trajectory-directory> --out <video.mp4>".into());
            }
            let (config, frames) = fluid::read_run(Path::new(&args[1]))?;
            fluid::recompute_energies(&config, &frames)?;
            let local_python = Path::new(env!("CARGO_MANIFEST_DIR")).join("../.venv/bin/python");
            let python = env::var_os("MD_PYTHON")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    if local_python.exists() {
                        local_python
                    } else {
                        PathBuf::from("python3")
                    }
                });
            let status = Command::new(&python)
                .arg("-c")
                .arg(include_str!("../../scripts/render_video.py"))
                .arg(&args[1])
                .arg("--out")
                .arg(&args[3])
                .status()
                .map_err(|e| format!("start video renderer {}: {e}", python.display()))?;
            if !status.success() {
                return Err("video renderer failed; install week2/requirements.txt into .venv or set MD_PYTHON".into());
            }
        }
        _ => return Err(format!("unknown command {}; use md --help", args[0])),
    }
    Ok(())
}

fn main() {
    if let Err(error) = execute() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
