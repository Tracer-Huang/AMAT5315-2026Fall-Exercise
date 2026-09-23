use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::{
    f64::consts::TAU,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};
use week4_fluid::{flow::*, io::*, Spectral};
fn create(path: PathBuf) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}

fn add_tracers(
    input: &Initial,
    args: &[String],
    out: &Path,
    wanted: &serde_json::Value,
) -> Result<()> {
    let previous: serde_json::Value = serde_json::from_reader(File::open(out.join("run.json"))?)?;
    for key in [
        "case",
        "n",
        "seed",
        "k_band",
        "nu",
        "dt",
        "t_end",
        "snapshot_every",
    ] {
        if previous.get(key) != wanted.get(key) {
            return Err(
                format!("existing run differs in {key}; choose a fresh output directory").into(),
            );
        }
    }
    let exists = out.join("tracers.jsonl").exists();
    if exists {
        for key in ["tracers", "tracer_seed", "tracer_every"] {
            if previous.get(key) != wanted.get(key) {
                return Err(format!(
                    "existing tracers differ in {key}; choose a fresh output directory"
                )
                .into());
            }
        }
    }
    // Preserve each replay. The original velocity/vorticity recording is never replaced.
    let stage = out.join("tracer-replays").join(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_nanos()
            .to_string(),
    );
    let mut replay_args = args.to_vec();
    let output_flag = replay_args
        .iter()
        .position(|arg| arg == "--out")
        .ok_or("missing --out")?;
    replay_args[output_flag + 1] = stage.to_string_lossy().into_owned();
    let mut child = Command::new(std::env::current_exe()?)
        .args(replay_args)
        .stdin(Stdio::piped())
        .spawn()?;
    {
        let mut stdin = child.stdin.take().ok_or("missing replay stdin")?;
        serde_json::to_writer(&mut stdin, input)?;
    }
    if !child.wait()?.success() {
        return Err("tracer replay failed; original run preserved".into());
    }
    if fs::read(stage.join("fields.jsonl"))? != fs::read(out.join("fields.jsonl"))? {
        return Err("tracer replay fields differ from existing run; original run preserved".into());
    }
    if exists {
        if fs::read(stage.join("tracers.jsonl"))? != fs::read(out.join("tracers.jsonl"))? {
            return Err(
                "tracer replay differs from existing positions; original run preserved".into(),
            );
        }
        return Ok(());
    }
    let mut backup = create(out.join("run.before-tracers.json"))?;
    backup.write_all(&fs::read(out.join("run.json"))?)?;
    backup.flush()?;
    // Same-filesystem hard link publishes the new particle file without overwriting.
    fs::hard_link(stage.join("tracers.jsonl"), out.join("tracers.jsonl"))?;
    let pending = out.join("run.with-tracers.json");
    let mut metadata = create(pending.clone())?;
    metadata.write_all(&fs::read(stage.join("run.json"))?)?;
    metadata.flush()?;
    fs::rename(pending, out.join("run.json"))?;
    Ok(())
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args == ["--help"] {
        println!("fluid --nu NU --dt DT --t-end T --every INTERVAL --out DIRECTORY [--tracers COUNT --tracer-seed SEED --tracer-every STRIDE]\nReads field JSON on stdin. Existing runs only accept matching tracer replays; original fields are preserved.");
        return Ok(());
    }
    let map = flags(
        args.iter().cloned(),
        &[
            "--nu",
            "--dt",
            "--t-end",
            "--every",
            "--out",
            "--tracers",
            "--tracer-seed",
            "--tracer-every",
        ],
    )?;
    let nu = required(&map, "--nu")?;
    positive(nu, "nu", true)?;
    let dt = required(&map, "--dt")?;
    positive(dt, "dt", false)?;
    let end = required(&map, "--t-end")?;
    positive(end, "t-end", false)?;
    let every = required(&map, "--every")?;
    positive(every, "every", false)?;
    let out: PathBuf = required::<String>(&map, "--out")?.into();
    let count: usize = optional(&map, "--tracers", 0)?;
    let stride: usize = optional(&map, "--tracer-every", 1)?;
    if stride == 0 {
        return Err("tracer-every must be positive".into());
    }
    let tracer_seed: Option<u64> = if count > 0 {
        Some(required(&map, "--tracer-seed")?)
    } else {
        None
    };
    let input: Initial = serde_json::from_reader(std::io::stdin().lock())?;
    valid_n(input.n)?;
    let n = input.n;
    if input.u.len() != n * n
        || input.v.len() != n * n
        || !input.u.iter().chain(&input.v).all(|x| x.is_finite())
    {
        return Err("input u and v must be finite n*n arrays".into());
    }
    match input.case.as_str() {
        "taylor-green" if input.seed.is_none() && input.k_band.is_none() => (),
        "random" if input.seed.is_some() && input.k_band.is_some() => (),
        _ => return Err("invalid case / seed / k_band metadata".into()),
    }
    let metadata = serde_json::json!({"case":input.case,"n":n,"seed":input.seed,"k_band":input.k_band,"nu":nu,"dt":dt,"t_end":end,"snapshot_every":every,"tracers":count,"tracer_seed":tracer_seed,"tracer_every":stride,"method":"Fourier pseudospectral, 2/3 filter, coupled classical RK4","rng":"rand_chacha ChaCha8Rng; physical half-plane ky ascending then kx ascending"});
    if out.exists() && fs::read_dir(&out)?.next().is_some() {
        if count > 0 {
            return add_tracers(&input, &args, &out, &metadata);
        }
        return Err(format!(
            "output directory is not empty: {} (choose a fresh run folder)",
            out.display()
        )
        .into());
    }
    let s = Spectral::new(n);
    let mut w = vorticity(&s, &input.u, &input.v);
    s.filter(&mut w);
    let mut rng = ChaCha8Rng::seed_from_u64(tracer_seed.unwrap_or(0));
    let mut p: Vec<[f64; 2]> = (0..count)
        .map(|_| [rng.gen::<f64>() * TAU, rng.gen::<f64>() * TAU])
        .collect();
    fs::create_dir_all(&out)?;
    let mut meta = create(out.join("run.json"))?;
    serde_json::to_writer_pretty(&mut meta, &metadata)?;
    writeln!(meta)?;
    meta.flush()?;
    let mut frames = create(out.join("fields.jsonl"))?;
    let mut tracer_file = if count > 0 {
        Some(create(out.join("tracers.jsonl"))?)
    } else {
        None
    };
    let mut stdout = BufWriter::new(std::io::stdout().lock());
    writeln!(stdout, "t\tE\tZ")?;
    let mut t = 0.;
    let mut steps = 0usize;
    let mut snapshot = 0usize;
    loop {
        let (u, v) = s.velocity(&w);
        let omega = s.inverse(&w);
        let e = energy(&u, &v);
        let z = omega.iter().map(|a| a * a).sum::<f64>() / (2. * (n * n) as f64);
        if !e.is_finite() || !z.is_finite() {
            return Err("unstable/nonfinite solution; reduce dt".into());
        }
        writeln!(stdout, "{t:.6}\t{e:.6}\t{z:.6}")?;
        write!(frames, "{{\"t\":{t:.12},\"step\":{steps},\"u\":")?;
        array(&mut frames, u.into_iter())?;
        write!(frames, ",\"v\":")?;
        array(&mut frames, v.into_iter())?;
        write!(frames, ",\"omega\":")?;
        array(&mut frames, omega.into_iter())?;
        writeln!(frames, "}}")?;
        if let Some(file) = tracer_file.as_mut() {
            if snapshot.is_multiple_of(stride) {
                write!(file, "{{\"t\":{t:.12},\"x\":")?;
                array(file, p.iter().map(|q| q[0]))?;
                write!(file, ",\"y\":")?;
                array(file, p.iter().map(|q| q[1]))?;
                writeln!(file, "}}")?;
            }
        }
        if t >= end {
            break;
        }
        snapshot += 1;
        let target = (snapshot as f64 * every).min(end);
        while target - t > 1e-13 * end.max(1.) {
            let h = dt.min(target - t);
            if h * nu * 2. * ((n / 3) as f64).powi(2) > 2.78 {
                return Err("dt exceeds RK4 viscous stability bound".into());
            }
            step(&s, &mut w, &mut p, nu, h);
            t += h;
            steps += 1;
        }
        t = target;
    }
    frames.flush()?;
    if let Some(mut f) = tracer_file {
        f.flush()?;
    }
    stdout.flush()?;
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("fluid: {e}");
        std::process::exit(1);
    }
}
