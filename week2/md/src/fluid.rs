use crate::dynamics::{MdResult, System, Vector, VelocityVerlet, advance};
use crate::periodic;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub n: usize,
    pub rho: f64,
    #[serde(rename = "box")]
    pub box_size: Vector,
    pub dt: f64,
    pub temperature: f64,
    pub eq_steps: usize,
    pub steps: usize,
    pub sample_every: usize,
    pub seed: u64,
    pub integrator: String,
    #[serde(default)]
    pub force: periodic::ForceMethod,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ramp_to: Option<f64>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            n: 100,
            rho: 0.8,
            box_size: [12.01405707067377, 10.404478625719542],
            dt: 0.01,
            temperature: 0.5,
            eq_steps: 2000,
            steps: 10000,
            sample_every: 50,
            seed: 2026,
            integrator: "velocity-verlet".into(),
            force: periodic::ForceMethod::Cells,
            ramp_to: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    pub step: usize,
    pub t: f64,
    pub pos: Vec<Vector>,
    pub vel: Vec<Vector>,
    #[serde(rename = "E_pot")]
    pub e_pot: f64,
    #[serde(rename = "E_kin")]
    pub e_kin: f64,
}

#[derive(Debug)]
pub struct CheckReport {
    pub secular_drift: f64,
    pub t_speed: f64,
    pub chi2_per_dof: f64,
    pub passed: bool,
}

impl RunConfig {
    pub fn validate(&self) -> MdResult<()> {
        let (_, expected_box) = periodic::lattice(self.n, self.rho)?;
        for axis in 0..2 {
            if !self.box_size[axis].is_finite()
                || (self.box_size[axis] - expected_box[axis]).abs() > 1e-10 * expected_box[axis]
            {
                return Err("box dimensions do not match n and rho".into());
            }
        }
        if !self.dt.is_finite()
            || self.dt <= 0.0
            || !self.temperature.is_finite()
            || self.temperature <= 0.0
        {
            return Err("dt and temperature must be finite and positive".into());
        }
        if self.steps == 0 || self.sample_every == 0 || self.sample_every > self.steps {
            return Err(
                "steps and sample-every must be positive, with sample-every <= steps".into(),
            );
        }
        if !(self.steps as f64 * self.dt).is_finite() {
            return Err("production duration must be finite".into());
        }
        ramp_target(self, 0)?;
        if self.integrator != "velocity-verlet" {
            return Err("the equilibrium CLI requires velocity-verlet; Euler remains in the dimer experiment".into());
        }
        Ok(())
    }
}

fn initialize(config: &RunConfig) -> MdResult<System> {
    let (positions, box_size) = periodic::lattice(config.n, config.rho)?;
    let normal = Normal::new(0.0, config.temperature.sqrt()).map_err(|e| e.to_string())?;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut velocities: Vec<Vector> = (0..config.n)
        .map(|_| [normal.sample(&mut rng), normal.sample(&mut rng)])
        .collect();
    let mean = [
        velocities.iter().map(|v| v[0]).sum::<f64>() / config.n as f64,
        velocities.iter().map(|v| v[1]).sum::<f64>() / config.n as f64,
    ];
    for v in &mut velocities {
        for axis in 0..2 {
            v[axis] -= mean[axis];
        }
    }
    let mut system = System::periodic_with_method(positions, velocities, box_size, config.force)?;
    rescale(&mut system, config.temperature)?;
    Ok(system)
}

pub fn rescale(system: &mut System, target: f64) -> MdResult<()> {
    let t = 2.0 * system.kinetic_energy() / (2 * system.velocities.len() - 2) as f64;
    if !t.is_finite() || t <= 0.0 || !target.is_finite() || target <= 0.0 {
        return Err("thermostat requires finite positive current and target temperatures".into());
    }
    let factor = (target / t).sqrt();
    for v in &mut system.velocities {
        for x in v {
            *x *= factor;
        }
    }
    Ok(())
}

pub fn ramp_target(config: &RunConfig, step: usize) -> MdResult<f64> {
    if config.steps == 0 || step > config.steps {
        return Err("heating step is outside production".into());
    }
    let end = config.ramp_to.unwrap_or(config.temperature);
    if !config.temperature.is_finite()
        || config.temperature <= 0.0
        || !end.is_finite()
        || end < config.temperature
    {
        return Err("ramp-to must be finite and at least the positive initial temperature".into());
    }
    if step == 0 {
        return Ok(config.temperature);
    }
    if step == config.steps {
        return Ok(end);
    }
    Ok(config.temperature + (end - config.temperature) * step as f64 / config.steps as f64)
}

pub fn simulate(
    config: &RunConfig,
    mut save: impl FnMut(&Frame) -> MdResult<()>,
) -> MdResult<usize> {
    config.validate()?;
    let mut system = initialize(config)?;
    for step in 1..=config.eq_steps {
        advance(&VelocityVerlet, &mut system, config.dt)?;
        if step % 50 == 0 {
            rescale(&mut system, config.temperature)?;
        }
    }
    let mut count = 0;
    // Ordinary production is NVE. An explicit ramp intentionally adds energy.
    for step in 1..=config.steps {
        advance(&VelocityVerlet, &mut system, config.dt)?;
        if config.ramp_to.is_some() && (step % 50 == 0 || step == config.steps) {
            rescale(&mut system, ramp_target(config, step)?)?;
        }
        if step % config.sample_every == 0 {
            let frame = Frame {
                step,
                t: step as f64 * config.dt,
                pos: system.positions.clone(),
                vel: system.velocities.clone(),
                e_pot: system.potential_energy,
                e_kin: system.kinetic_energy(),
            };
            if !frame.e_kin.is_finite() {
                return Err(format!("non-finite kinetic energy at step {step}"));
            }
            save(&frame)?;
            count += 1;
        }
    }
    Ok(count)
}

pub fn simulate_frames(config: &RunConfig) -> MdResult<Vec<Frame>> {
    let mut frames = Vec::new();
    simulate(config, |frame| {
        frames.push(frame.clone());
        Ok(())
    })?;
    Ok(frames)
}

fn validate_frames(config: &RunConfig, frames: &[Frame]) -> MdResult<()> {
    config.validate()?;
    let expected = config.steps / config.sample_every;
    if frames.len() != expected || frames.is_empty() {
        return Err(format!(
            "expected {expected} frames, found {}",
            frames.len()
        ));
    }
    for (index, frame) in frames.iter().enumerate() {
        let step = (index + 1) * config.sample_every;
        if frame.step != step
            || !frame.t.is_finite()
            || (frame.t - step as f64 * config.dt).abs()
                > 1e-10 * (step as f64 * config.dt).max(1.0)
        {
            return Err(format!("invalid step/time in frame {index}"));
        }
        if frame.pos.len() != config.n || frame.vel.len() != config.n {
            return Err(format!("particle array length mismatch in frame {index}"));
        }
        if frame
            .pos
            .iter()
            .chain(&frame.vel)
            .flatten()
            .any(|x| !x.is_finite())
            || !frame.e_pot.is_finite()
            || !frame.e_kin.is_finite()
        {
            return Err(format!("non-finite data in frame {index}"));
        }
        for p in &frame.pos {
            for axis in 0..2 {
                if p[axis] < 0.0 || p[axis] >= config.box_size[axis] {
                    return Err(format!("position outside periodic box in frame {index}"));
                }
            }
        }
    }
    Ok(())
}

/// Recompute every energy from raw positions and velocities, then cross-check stored values.
pub fn recompute_energies(config: &RunConfig, frames: &[Frame]) -> MdResult<Vec<f64>> {
    validate_frames(config, frames)?;
    let mut forces = vec![[0.0; 2]; config.n];
    let mut energies = Vec::with_capacity(frames.len());
    for (index, frame) in frames.iter().enumerate() {
        let potential = periodic::forces(&frame.pos, config.box_size, &mut forces)?;
        let kinetic = 0.5
            * frame
                .vel
                .iter()
                .map(|v| v[0] * v[0] + v[1] * v[1])
                .sum::<f64>();
        for (name, recomputed, stored) in [
            ("potential", potential, frame.e_pot),
            ("kinetic", kinetic, frame.e_kin),
        ] {
            if !recomputed.is_finite()
                || (recomputed - stored).abs() > 1e-8 * recomputed.abs().max(1.0)
            {
                return Err(format!(
                    "stored {name} energy disagrees with raw data in frame {index}: {stored} vs {recomputed}"
                ));
            }
        }
        energies.push(potential + kinetic);
    }
    Ok(energies)
}

pub fn analyze(config: &RunConfig, frames: &[Frame]) -> MdResult<CheckReport> {
    if config.ramp_to.is_some() {
        return Err("heated trajectories intentionally exchange energy; use the unheated default run for the contract check".into());
    }
    let energies = recompute_energies(config, frames)?;
    let initial = energies[0].abs();
    if initial < 1e-12 {
        return Err("first saved energy is too close to zero to normalize drift".into());
    }
    let k = (frames.len() / 10).max(1);
    let first = energies[..k].iter().sum::<f64>() / k as f64;
    let last = energies[energies.len() - k..].iter().sum::<f64>() / k as f64;
    let secular_drift = (last - first).abs() / initial;
    let speeds: Vec<f64> = frames
        .iter()
        .flat_map(|f| f.vel.iter().map(|v| v[0].hypot(v[1])))
        .collect();
    let t_speed = speeds.iter().map(|v| v * v).sum::<f64>() / (2 * speeds.len()) as f64;
    if !t_speed.is_finite() || t_speed <= 0.0 {
        return Err("speed temperature must be finite and positive".into());
    }
    let mut edges = Vec::with_capacity(25);
    for b in 0..24 {
        edges.push((-2.0 * t_speed * (1.0 - b as f64 / 24.0).ln()).sqrt());
    }
    edges.push(f64::INFINITY);
    let mut counts = [0usize; 24];
    for v in &speeds {
        let bin = edges
            .partition_point(|edge| v >= edge)
            .saturating_sub(1)
            .min(23);
        counts[bin] += 1;
    }
    let expected = speeds.len() as f64 / 24.0;
    let chi2_per_dof = counts
        .iter()
        .map(|&count| (count as f64 - expected).powi(2) / expected)
        .sum::<f64>()
        / 22.0;
    let passed = secular_drift < 2e-3 && (t_speed - 0.5).abs() < 0.05 && chi2_per_dof < 2.0;
    Ok(CheckReport {
        secular_drift,
        t_speed,
        chi2_per_dof,
        passed,
    })
}

pub fn write_run(config: &RunConfig, out: &Path) -> MdResult<usize> {
    config.validate()?;
    fs::create_dir_all(out).map_err(|e| format!("create output directory: {e}"))?;
    let stamp = format!(
        "{}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos(),
        std::process::id()
    );
    let staged_traj = out.join(format!(".traj-{stamp}.jsonl"));
    let staged_run = out.join(format!(".run-{stamp}.json"));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged_traj)
        .map_err(|e| e.to_string())?;
    let mut writer = BufWriter::new(file);
    let count = simulate(config, |frame| {
        serde_json::to_writer(&mut writer, frame).map_err(|e| e.to_string())?;
        writeln!(writer).map_err(|e| e.to_string())
    })?;
    writer.flush().map_err(|e| e.to_string())?;
    let mut metadata = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staged_run)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut metadata, config).map_err(|e| e.to_string())?;
    writeln!(metadata).map_err(|e| e.to_string())?;
    // Preserve earlier generated outputs before installing the completed new run.
    if out.join("run.json").exists() || out.join("traj.jsonl").exists() {
        let history = out.join("history").join(&stamp);
        fs::create_dir_all(&history).map_err(|e| e.to_string())?;
        for name in ["run.json", "traj.jsonl"] {
            if out.join(name).exists() {
                fs::copy(out.join(name), history.join(name)).map_err(|e| e.to_string())?;
            }
        }
        eprintln!("preserved previous output in {}", history.display());
    }
    fs::rename(staged_run, out.join("run.json")).map_err(|e| e.to_string())?;
    fs::rename(staged_traj, out.join("traj.jsonl")).map_err(|e| e.to_string())?;
    Ok(count)
}

pub fn read_run(path: &Path) -> MdResult<(RunConfig, Vec<Frame>)> {
    let config: RunConfig = serde_json::from_reader(BufReader::new(
        File::open(path.join("run.json")).map_err(|e| format!("read run.json: {e}"))?,
    ))
    .map_err(|e| format!("parse run.json: {e}"))?;
    config.validate()?;
    let file = File::open(path.join("traj.jsonl")).map_err(|e| format!("read traj.jsonl: {e}"))?;
    let mut frames = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        let frame: Frame = serde_json::from_str(&line)
            .map_err(|e| format!("parse traj.jsonl line {}: {e}", index + 1))?;
        frames.push(frame);
        if frames.len() > config.steps / config.sample_every {
            return Err("too many trajectory frames".into());
        }
    }
    validate_frames(&config, &frames)?;
    Ok((config, frames))
}
