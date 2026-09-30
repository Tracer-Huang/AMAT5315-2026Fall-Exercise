use seismic::{Action, Model, schedule};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[derive(Deserialize)]
struct Experiment {
    name: String,
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
    steps: usize,
    source_frequency: f64,
    source_peak_time: f64,
    source_amplitude: f64,
    sponge_width: usize,
    sponge_strength: f64,
    shots: Vec<[f64; 2]>,
    receivers: Vec<[usize; 2]>,
    background: Vec<Vec<f64>>,
    perturbation: Vec<Vec<f64>>,
}
impl Experiment {
    fn validate(&self) -> Result<()> {
        if self.nx < 3
            || self.nz < 3
            || self.steps == 0
            || self.dx <= 0.
            || self.dt <= 0.
            || self.shots.is_empty()
            || self.receivers.is_empty()
        {
            return Err("invalid dimensions, timestep or acquisition".into());
        }
        for a in [&self.background, &self.perturbation] {
            if a.len() != self.nz
                || a.iter()
                    .any(|r| r.len() != self.nx || r.iter().any(|v| !v.is_finite()))
            {
                return Err("invalid model shape or values".into());
            }
        }
        let max = self.background.iter().flatten().copied().fold(0., f64::max);
        if self.background.iter().flatten().any(|v| *v <= 0.)
            || max * self.dt / self.dx > 1. / 2f64.sqrt()
        {
            return Err("nonpositive wave speed or unstable 2D Courant number".into());
        }
        if self
            .receivers
            .iter()
            .any(|r| r[0] >= self.nx || r[1] >= self.nz)
            || self.shots.iter().any(|p| {
                p[0] < 0. || p[1] < 0. || p[0] > (self.nx - 1) as f64 || p[1] > (self.nz - 1) as f64
            })
        {
            return Err("acquisition outside grid".into());
        }
        Ok(())
    }
    fn model(&self) -> Model {
        let mut sigma = vec![0.; self.nx * self.nz];
        if self.sponge_width > 0 {
            for z in 0..self.nz {
                for x in 0..self.nx {
                    let distance = x.min(z).min(self.nx - 1 - x).min(self.nz - 1 - z) as f64;
                    let f = (1. - distance / self.sponge_width as f64).max(0.);
                    sigma[z * self.nx + x] = self.sponge_strength * f * f;
                }
            }
        }
        Model {
            nx: self.nx,
            nz: self.nz,
            dx: self.dx,
            dt: self.dt,
            c: self.background.iter().flatten().copied().collect(),
            sigma,
        }
    }
    fn footprint(&self, shot: usize) -> Vec<f64> {
        let p = self.shots[shot];
        (0..self.nx * self.nz)
            .map(|i| {
                let x = (i % self.nx) as f64 - p[0];
                let z = (i / self.nx) as f64 - p[1];
                (-(x * x + z * z) / 2.).exp()
            })
            .collect()
    }
    fn source(&self, n: usize, foot: &[f64], q: &mut [f64]) {
        let theta = std::f64::consts::PI
            * self.source_frequency
            * (n as f64 * self.dt - self.source_peak_time);
        let pulse = self.source_amplitude * (1. - 2. * theta * theta) * (-theta * theta).exp();
        for (a, b) in q.iter_mut().zip(foot) {
            *a = pulse * b;
        }
    }
}
fn write_json(p: &Path, v: &Value) -> Result<()> {
    fs::write(p, serde_json::to_string_pretty(v)? + "\n")?;
    Ok(())
}
fn npy_write(path: &Path, data: &[f64], shape: &[usize], f32_data: bool) -> Result<()> {
    if shape.iter().product::<usize>() != data.len() {
        return Err("NPY shape mismatch".into());
    }
    let dtype = if f32_data { "<f4" } else { "<f8" };
    let dims = shape
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let mut h = format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': ({dims},), }}");
    let pad = (64 - (10 + h.len() + 1) % 64) % 64;
    h.push_str(&" ".repeat(pad));
    h.push('\n');
    let mut f = BufWriter::new(fs::File::create(path)?);
    f.write_all(b"\x93NUMPY\x01\x00")?;
    f.write_all(&(h.len() as u16).to_le_bytes())?;
    f.write_all(h.as_bytes())?;
    for x in data {
        if !x.is_finite() {
            return Err("nonfinite result".into());
        }
        if f32_data {
            f.write_all(&(*x as f32).to_le_bytes())?;
        } else {
            f.write_all(&x.to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(())
}
fn npy_read(path: &Path, shape: &[usize]) -> Result<Vec<f64>> {
    let b = fs::read(path)?;
    if b.len() < 10 || &b[..8] != b"\x93NUMPY\x01\x00" {
        return Err("requires NPY v1 float64 C-order".into());
    }
    let len = u16::from_le_bytes([b[8], b[9]]) as usize;
    if b.len() < 10 + len {
        return Err("truncated NPY header".into());
    }
    let h = std::str::from_utf8(&b[10..10 + len])?;
    if !h.contains("'<f8'") || !h.contains("'fortran_order': False") {
        return Err("requires little-endian float64 C-order".into());
    }
    let start = h.find('(').ok_or("missing NPY shape")?;
    let end = h.find(')').ok_or("missing NPY shape")?;
    let dims = h[start + 1..end]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse())
        .collect::<std::result::Result<Vec<usize>, _>>()?;
    if dims != shape || b.len() != 10 + len + shape.iter().product::<usize>() * 8 {
        return Err("NPY shape or byte length mismatch".into());
    }
    let values = b[10 + len..]
        .chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite receiver data".into());
    }
    Ok(values)
}
fn norm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let argv = std::env::args().skip(1).collect::<Vec<_>>();
    if argv == ["--help"] {
        println!(
            "seismic --experiment FILE --mode forward|born|adjoint --out DIR [--data NPY] [--every STRIDE] [--storage full|treeverse --checkpoints BUDGET]\nExisting nonempty outputs are never overwritten."
        );
        return Ok(());
    }
    if argv == ["--enzyme-smoke"] {
        println!("{:?}", seismic::cube_probe(2.));
        return Ok(());
    }
    let mut args = BTreeMap::new();
    if argv.len() % 2 != 0 {
        return Err("options require values; see --help".into());
    }
    for pair in argv.chunks_exact(2) {
        if ![
            "--experiment",
            "--mode",
            "--out",
            "--data",
            "--every",
            "--storage",
            "--checkpoints",
        ]
        .contains(&pair[0].as_str())
            || args.insert(pair[0].clone(), pair[1].clone()).is_some()
        {
            return Err("unknown or duplicate option".into());
        }
    }
    let experiment = args.get("--experiment").ok_or("--experiment is required")?;
    let mode = args.get("--mode").ok_or("--mode is required")?.as_str();
    if !["forward", "born", "adjoint"].contains(&mode) {
        return Err("invalid mode".into());
    }
    let out = Path::new(args.get("--out").ok_or("--out is required")?);
    let every = args
        .get("--every")
        .map(|x| x.parse::<usize>())
        .transpose()?;
    if every == Some(0) {
        return Err("--every must be positive".into());
    }
    if every.is_some() && mode == "born" {
        return Err("recording is defined for forward or adjoint".into());
    }
    let storage = args.get("--storage").map(String::as_str).unwrap_or("full");
    if !["full", "treeverse"].contains(&storage) {
        return Err("invalid storage".into());
    }
    let budget = args
        .get("--checkpoints")
        .map(|x| x.parse::<usize>())
        .transpose()?;
    if storage == "treeverse" && (mode != "adjoint" || budget.is_none_or(|b| b == 0)) {
        return Err("treeverse requires adjoint and positive --checkpoints".into());
    }
    if storage == "full" && budget.is_some() {
        return Err("--checkpoints requires treeverse".into());
    }
    let bytes = fs::read(experiment)?;
    let e: Experiment = serde_json::from_slice(&bytes)?;
    e.validate()?;
    if mode == "adjoint" && storage == "full" && e.name == "marmousi" {
        return Err("Marmousi full history prohibited; use treeverse".into());
    }
    let data = if mode == "adjoint" {
        Some(npy_read(
            Path::new(args.get("--data").ok_or("adjoint requires --data")?),
            &[e.shots.len(), e.steps, e.receivers.len()],
        )?)
    } else {
        if args.contains_key("--data") {
            return Err("--data is only for adjoint".into());
        }
        None
    };
    if out.exists() && fs::read_dir(out)?.next().is_some() {
        return Err("output directory is nonempty; choose a new versioned directory".into());
    }
    fs::create_dir_all(out)?;
    let started = Instant::now();
    let model = e.model();
    let m = model.size();
    let nr = e.receivers.len();
    let mut image = vec![0.; m];
    let mut traces = Vec::new();
    let mut frames = Vec::new();
    let mut frame_steps = Vec::new();
    let mut per_shot = vec![];
    println!("shot\tmode\tl2");
    for shot in 0..e.shots.len() {
        let footprint = e.footprint(shot);
        let mut q = vec![0.; m];
        let mut state = vec![0.; 2 * m];
        let mut next = vec![0.; 2 * m];
        if mode != "adjoint" {
            let mut ds = vec![0.; 2 * m];
            let mut dn = vec![0.; 2 * m];
            let dc = e.perturbation.iter().flatten().copied().collect::<Vec<_>>();
            let begin = traces.len();
            for n in 0..e.steps {
                e.source(n, &footprint, &mut q);
                if mode == "born" {
                    model.jvp(&state, &ds, &dc, &q, &mut next, &mut dn);
                } else {
                    model.forward(&state, &q, &mut next);
                }
                let values = if mode == "born" { &dn } else { &next };
                for r in &e.receivers {
                    traces.push(values[m + r[1] * e.nx + r[0]]);
                }
                std::mem::swap(&mut state, &mut next);
                if mode == "born" {
                    std::mem::swap(&mut ds, &mut dn);
                }
                if shot == 0 && every.is_some_and(|k| (n + 1) % k == 0) {
                    frames.extend_from_slice(&state[m..]);
                    frame_steps.push(n + 1);
                }
            }
            println!("{shot}\t{mode}\t{:.12e}", norm(&traces[begin..]));
        } else {
            let weights = &data.as_ref().unwrap()[shot * e.steps * nr..(shot + 1) * e.steps * nr];
            let mut adj = vec![0.; 2 * m];
            let mut shot_image = vec![0.; m];
            let mut calls = 0;
            let mut reverse = 0;
            let mut peak = 1;
            let mut events = vec![];
            let mut saved = BTreeMap::from([(0, state.clone())]);
            let actions = if storage == "full" {
                for n in 0..e.steps {
                    e.source(n, &footprint, &mut q);
                    model.forward(&state, &q, &mut next);
                    std::mem::swap(&mut state, &mut next);
                    saved.insert(n + 1, state.clone());
                    calls += 1;
                }
                peak = saved.len();
                (0..e.steps).rev().map(Action::Grad).collect::<Vec<_>>()
            } else {
                schedule(e.steps, budget.unwrap())
            };
            for action in actions {
                let (kind, n) = match action {
                    Action::Restore(n) => {
                        state.copy_from_slice(saved.get(&n).ok_or("restore missing checkpoint")?);
                        ("restore", n)
                    }
                    Action::Call(n) => {
                        e.source(n, &footprint, &mut q);
                        model.forward(&state, &q, &mut next);
                        std::mem::swap(&mut state, &mut next);
                        calls += 1;
                        ("call", n)
                    }
                    Action::Store(n) => {
                        if saved.insert(n, state.clone()).is_some() {
                            return Err("duplicate checkpoint".into());
                        }
                        peak = peak.max(saved.len());
                        if saved.len() > budget.unwrap() + 1 {
                            return Err("checkpoint budget exceeded".into());
                        }
                        ("store", n)
                    }
                    Action::Fetch(n) => {
                        if n == 0 || saved.remove(&n).is_none() {
                            return Err("invalid checkpoint discard".into());
                        }
                        ("fetch", n)
                    }
                    Action::Grad(n) => {
                        if n != e.steps - 1 - reverse {
                            return Err("reverse steps out of order".into());
                        }
                        for (k, r) in e.receivers.iter().enumerate() {
                            adj[m + r[1] * e.nx + r[0]] += weights[n * nr + k];
                        }
                        e.source(n, &footprint, &mut q);
                        let input = saved.get(&n).ok_or("reverse needs saved state")?;
                        let (mut ds, dc) = model.vjp(input, &q, &adj);
                        // Outer states are constrained to zero; discard sensitivities outside the evolving domain.
                        for z in 0..e.nz {
                            for x in 0..e.nx {
                                if x == 0 || z == 0 || x + 1 == e.nx || z + 1 == e.nz {
                                    let i = z * e.nx + x;
                                    ds[i] = 0.;
                                    ds[m + i] = 0.;
                                }
                            }
                        }
                        adj = ds;
                        for (i, v) in shot_image.iter_mut().zip(dc) {
                            *i += v;
                        }
                        reverse += 1;
                        if shot == 0 && every.is_some_and(|k| n % k == 0) {
                            frames.extend_from_slice(&adj[m..]);
                            frame_steps.push(n);
                        }
                        ("grad", n)
                    }
                };
                if storage == "treeverse" {
                    events.push(json!({"action":kind,"step":n,"saved_states":saved.len()}));
                }
            }
            for (a, b) in image.iter_mut().zip(&shot_image) {
                *a += b;
            }
            let mut counters = json!({"shot":shot,"scheduler_forward_calls":calls,"reverse_calls":reverse,"peak_saved_states":peak});
            if storage == "treeverse" {
                let filename = format!("actions-{shot}.json");
                write_json(&out.join(&filename), &json!(events))?;
                counters["actions_file"] = json!(filename);
            }
            per_shot.push(counters);
            println!("{shot}\t{mode}\t{:.12e}", norm(&shot_image));
        }
    }
    if mode == "adjoint" {
        npy_write(&out.join("image.npy"), &image, &[e.nz, e.nx], false)?;
    } else {
        npy_write(
            &out.join(if mode == "born" {
                "born_data.npy"
            } else {
                "traces.npy"
            }),
            &traces,
            &[e.shots.len(), e.steps, nr],
            false,
        )?;
    }
    let mut input: Value = serde_json::from_slice(&bytes)?;
    input.as_object_mut().unwrap().remove("background");
    input.as_object_mut().unwrap().remove("perturbation");
    let mut run = json!({"experiment":input,"experiment_file":experiment,"experiment_sha256":format!("{:x}",Sha256::digest(&bytes)),"arguments":args,"precision":"float64","parameter":"velocity","inner_product":"unweighted Euclidean sums"});
    if let Some(stride) = every {
        npy_write(
            &out.join("wavefield.npy"),
            &frames,
            &[frame_steps.len(), e.nz, e.nx],
            true,
        )?;
        run["recording"] = json!({"every":stride,"steps":frame_steps,"times":frame_steps.iter().map(|n|*n as f64*e.dt).collect::<Vec<_>>()});
    }
    write_json(&out.join("run.json"), &run)?;
    let stats = if mode == "adjoint" {
        let peak = per_shot
            .iter()
            .map(|s| s["peak_saved_states"].as_u64().unwrap())
            .max()
            .unwrap();
        json!({"storage":storage,"checkpoints":budget,"state_wavefields":2,"saved_state_bytes":2*m*8,"peak_saved_states":peak,"peak_saved_bytes":peak*2*m as u64*8,"scheduler_forward_calls":per_shot.iter().map(|s|s["scheduler_forward_calls"].as_u64().unwrap()).sum::<u64>(),"reverse_calls":per_shot.iter().map(|s|s["reverse_calls"].as_u64().unwrap()).sum::<u64>(),"per_shot":per_shot})
    } else {
        Value::Null
    };
    write_json(
        &out.join("result.json"),
        &json!({"schema":"week5-born-rtm-v2","experiment":e.name,"mode":mode,"nx":e.nx,"nz":e.nz,"steps":e.steps,"dt":e.dt,"dx":e.dx,"shots":e.shots,"receivers":e.receivers,"implementation":"Rust; genuine Enzyme timestep JVP/VJP; nightly-2026-09-05","seconds":started.elapsed().as_secs_f64(),"statistics":stats}),
    )?;
    Ok(())
}
