//! Particle ownership and interchangeable numerical integrators.
pub type Vector = [f64; 2];
pub type MdResult<T> = Result<T, String>;

#[derive(Clone, Debug)]
pub struct System {
    pub positions: Vec<Vector>,
    pub velocities: Vec<Vector>,
    pub accelerations: Vec<Vector>,
    pub potential_energy: f64,
}

impl System {
    pub fn new(positions: Vec<Vector>, velocities: Vec<Vector>) -> MdResult<Self> {
        if positions.is_empty() || positions.len() != velocities.len() {
            return Err("positions and velocities must have the same nonzero length".into());
        }
        if positions
            .iter()
            .chain(&velocities)
            .flatten()
            .any(|x| !x.is_finite())
        {
            return Err("particle coordinates and velocities must be finite".into());
        }
        let mut system = Self {
            accelerations: vec![[0.0; 2]; positions.len()],
            positions,
            velocities,
            potential_energy: 0.0,
        };
        system.update_forces()?;
        Ok(system)
    }

    pub fn total_energy(&self) -> f64 {
        self.kinetic_energy() + self.potential_energy
    }

    pub fn kinetic_energy(&self) -> f64 {
        0.5 * self
            .velocities
            .iter()
            .map(|v| v[0] * v[0] + v[1] * v[1])
            .sum::<f64>()
    }

    pub fn update_forces(&mut self) -> MdResult<()> {
        self.potential_energy = compute_forces(&self.positions, &mut self.accelerations)?;
        Ok(())
    }
}

/// Each unordered pair contributes equal and opposite forces exactly once.
#[inline(never)]
pub fn compute_forces(positions: &[Vector], forces: &mut [Vector]) -> MdResult<f64> {
    if positions.len() != forces.len() {
        return Err("force array length does not match particle count".into());
    }
    forces.fill([0.0; 2]);
    let mut potential = 0.0;
    for i in 0..positions.len() {
        for j in i + 1..positions.len() {
            let d = [
                positions[i][0] - positions[j][0],
                positions[i][1] - positions[j][1],
            ];
            let r = d[0].hypot(d[1]);
            if !r.is_finite() || r == 0.0 {
                return Err(format!("invalid distance between particles {i} and {j}"));
            }
            let scale = crate::pair::force(r) / r;
            potential += crate::pair::energy(r);
            for axis in 0..2 {
                let f = scale * d[axis];
                forces[i][axis] += f;
                forces[j][axis] -= f;
            }
        }
    }
    if !potential.is_finite() || forces.iter().flatten().any(|x| !x.is_finite()) {
        return Err("non-finite pair energy or force".into());
    }
    Ok(potential)
}

pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64) -> MdResult<()>;
}

pub struct ForwardEuler;
pub struct VelocityVerlet;

impl Integrator for ForwardEuler {
    fn step(&self, system: &mut System, dt: f64) -> MdResult<()> {
        validate_dt(dt)?;
        for ((x, v), a) in system
            .positions
            .iter_mut()
            .zip(&mut system.velocities)
            .zip(&system.accelerations)
        {
            for axis in 0..2 {
                x[axis] += dt * v[axis];
                v[axis] += dt * a[axis];
            }
        }
        system.update_forces()
    }
}
impl Integrator for VelocityVerlet {
    fn step(&self, system: &mut System, dt: f64) -> MdResult<()> {
        validate_dt(dt)?;
        for ((x, v), a) in system
            .positions
            .iter_mut()
            .zip(&mut system.velocities)
            .zip(&system.accelerations)
        {
            for axis in 0..2 {
                v[axis] += 0.5 * dt * a[axis];
                x[axis] += dt * v[axis];
            }
        }
        system.update_forces()?;
        for (v, a) in system.velocities.iter_mut().zip(&system.accelerations) {
            for axis in 0..2 {
                v[axis] += 0.5 * dt * a[axis];
            }
        }
        Ok(())
    }
}

pub fn advance(method: &impl Integrator, system: &mut System, dt: f64) -> MdResult<()> {
    method.step(system, dt)
}

fn validate_dt(dt: f64) -> MdResult<()> {
    if dt.is_finite() && dt > 0.0 {
        Ok(())
    } else {
        Err("timestep must be finite and positive".into())
    }
}
