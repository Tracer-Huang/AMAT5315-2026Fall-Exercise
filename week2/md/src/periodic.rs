use crate::dynamics::{MdResult, Vector};
pub const CUTOFF: f64 = 2.5;

#[derive(Clone, Debug)]
pub struct CellList;

impl CellList {
    pub fn new(_box_size: Vector) -> MdResult<Self> {
        todo!("Part 5: create a deduplicated periodic cell grid")
    }

    pub fn forces(&mut self, _positions: &[Vector], _output: &mut [Vector]) -> MdResult<f64> {
        todo!("Part 5: evaluate forces with the cell list")
    }
}

pub fn lattice(n: usize, rho: f64) -> MdResult<(Vec<Vector>, Vector)> {
    if n > 1_000_000 {
        return Err("n is too large for this teaching simulator".into());
    }
    let side = (n as f64).sqrt() as usize;
    if side < 2 || side.checked_mul(side) != Some(n) || side % 2 != 0 {
        return Err("n must be an even-sided square (for example 100, 400, 1600)".into());
    }
    if !rho.is_finite() || rho <= 0.0 {
        return Err("density must be finite and positive".into());
    }
    let a = (2.0 / (3.0_f64.sqrt() * rho)).sqrt();
    let h = 0.5 * 3.0_f64.sqrt() * a;
    let box_size = [side as f64 * a, side as f64 * h];
    validate_box(box_size)?;
    let mut positions = Vec::with_capacity(n);
    for row in 0..side {
        for col in 0..side {
            positions.push([(col as f64 + 0.5 * (row % 2) as f64) * a, row as f64 * h]);
        }
    }
    Ok((positions, box_size))
}
pub fn minimum_image(d: f64, length: f64) -> f64 {
    d - length * (d / length).round()
}

pub fn validate_box(box_size: Vector) -> MdResult<()> {
    if box_size
        .iter()
        .any(|&x| !x.is_finite() || x <= 2.0 * CUTOFF)
    {
        return Err("both box sides must be finite and greater than twice the cutoff".into());
    }
    Ok(())
}

/// Naive all-pairs path. Kept as a named function for sampling profiles.
#[inline(never)]
pub fn forces(positions: &[Vector], box_size: Vector, output: &mut [Vector]) -> MdResult<f64> {
    validate_box(box_size)?;
    if output.len() != positions.len() {
        return Err("force array length does not match positions".into());
    }
    output.fill([0.0; 2]);
    let shift = crate::pair::energy(CUTOFF);
    let mut potential = 0.0;
    for i in 0..positions.len() {
        for j in i + 1..positions.len() {
            let dx = minimum_image(positions[i][0] - positions[j][0], box_size[0]);
            let dy = minimum_image(positions[i][1] - positions[j][1], box_size[1]);
            let r2 = dx * dx + dy * dy;
            if !r2.is_finite() || r2 == 0.0 {
                return Err(format!("invalid minimum-image distance for pair {i},{j}"));
            }
            if r2 >= CUTOFF * CUTOFF {
                continue;
            }
            let r = r2.sqrt();
            let scale = crate::pair::force(r) / r;
            potential += crate::pair::energy(r) - shift;
            output[i][0] += scale * dx;
            output[i][1] += scale * dy;
            output[j][0] -= scale * dx;
            output[j][1] -= scale * dy;
        }
    }
    if !potential.is_finite() || output.iter().flatten().any(|x| !x.is_finite()) {
        return Err("non-finite periodic force or energy".into());
    }
    Ok(potential)
}
