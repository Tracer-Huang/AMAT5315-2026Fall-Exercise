use crate::dynamics::{MdResult, Vector};
use serde::{Deserialize, Serialize};
use std::path::Path;

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
}

impl Default for RunConfig {
    fn default() -> Self {
        Self { n: 100, rho: 0.8, box_size: [12.01405707067377, 10.404478625719542],
            dt: 0.01, temperature: 0.5, eq_steps: 2000, steps: 10000,
            sample_every: 50, seed: 2026, integrator: "velocity-verlet".into() }
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

pub fn simulate_frames(_config: &RunConfig) -> MdResult<Vec<Frame>> {
    todo!("Part 4: simulate equilibrium fluid")
}
pub fn analyze(_config: &RunConfig, _frames: &[Frame]) -> MdResult<CheckReport> {
    todo!("Part 4: recompute physics from raw frames")
}
pub fn write_run(_config: &RunConfig, _out: &Path) -> MdResult<usize> {
    todo!("Part 4: write trajectory")
}
pub fn read_run(_path: &Path) -> MdResult<(RunConfig, Vec<Frame>)> {
    todo!("Part 4: validate and read trajectory")
}
