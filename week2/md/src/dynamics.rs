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
    pub fn new(_positions: Vec<Vector>, _velocities: Vec<Vector>) -> MdResult<Self> {
        todo!("Part 3: initialize an isolated system")
    }

    pub fn total_energy(&self) -> f64 {
        todo!("Part 3: total energy")
    }
}

pub trait Integrator {
    fn step(&self, system: &mut System, dt: f64) -> MdResult<()>;
}

pub struct ForwardEuler;
pub struct VelocityVerlet;

impl Integrator for ForwardEuler {
    fn step(&self, _system: &mut System, _dt: f64) -> MdResult<()> {
        todo!("Part 3: forward Euler")
    }
}
impl Integrator for VelocityVerlet {
    fn step(&self, _system: &mut System, _dt: f64) -> MdResult<()> {
        todo!("Part 3: velocity-Verlet")
    }
}

pub fn advance(_method: &impl Integrator, _system: &mut System, _dt: f64) -> MdResult<()> {
    todo!("Part 3: shared trait driver")
}
