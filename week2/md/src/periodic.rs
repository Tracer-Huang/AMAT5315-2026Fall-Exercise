use crate::dynamics::{MdResult, Vector};
pub const CUTOFF: f64 = 2.5;

pub fn lattice(_n: usize, _rho: f64) -> MdResult<(Vec<Vector>, Vector)> {
    todo!("Part 4: periodic triangular lattice")
}
pub fn minimum_image(_d: f64, _length: f64) -> f64 {
    todo!("Part 4: minimum image")
}
pub fn forces(_positions: &[Vector], _box_size: Vector, _output: &mut [Vector]) -> MdResult<f64> {
    todo!("Part 4: periodic shifted-cutoff forces")
}
