use crate::dynamics::{MdResult, Vector};
use serde::{Deserialize, Serialize};
pub const CUTOFF: f64 = 2.5;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ForceMethod {
    #[default]
    Naive,
    Cells,
}

impl ForceMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::Cells => "cells",
        }
    }
}

impl std::str::FromStr for ForceMethod {
    type Err = String;
    fn from_str(value: &str) -> MdResult<Self> {
        match value {
            "naive" => Ok(Self::Naive),
            "cells" => Ok(Self::Cells),
            _ => Err("force must be naive or cells".into()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CellList {
    pub(crate) box_size: Vector,
    nx: usize,
    ny: usize,
    bins: Vec<Vec<usize>>,
    neighbours: Vec<Vec<usize>>,
    candidates: Vec<usize>,
}

impl CellList {
    pub fn new(box_size: Vector) -> MdResult<Self> {
        validate_box(box_size)?;
        let nx = (box_size[0] / CUTOFF).floor() as usize;
        let ny = (box_size[1] / CUTOFF).floor() as usize;
        let count = nx
            .checked_mul(ny)
            .filter(|&n| n <= 1_000_000)
            .ok_or_else(|| {
                "cell grid too large; increase density or use --force naive".to_string()
            })?;
        let mut neighbours = Vec::with_capacity(count);
        for cy in 0..ny {
            for cx in 0..nx {
                let mut cells = Vec::with_capacity(9);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let x = (cx as isize + dx).rem_euclid(nx as isize) as usize;
                        let y = (cy as isize + dy).rem_euclid(ny as isize) as usize;
                        let index = y * nx + x;
                        if !cells.contains(&index) {
                            cells.push(index);
                        }
                    }
                }
                neighbours.push(cells);
            }
        }
        Ok(Self {
            box_size,
            nx,
            ny,
            bins: vec![Vec::new(); count],
            neighbours,
            candidates: Vec::new(),
        })
    }

    fn index(&self, p: Vector) -> usize {
        let x = ((p[0].rem_euclid(self.box_size[0]) / self.box_size[0]) * self.nx as f64).floor()
            as usize;
        let y = ((p[1].rem_euclid(self.box_size[1]) / self.box_size[1]) * self.ny as f64).floor()
            as usize;
        y.min(self.ny - 1) * self.nx + x.min(self.nx - 1)
    }

    #[inline(never)]
    pub fn forces(&mut self, positions: &[Vector], output: &mut [Vector]) -> MdResult<f64> {
        if output.len() != positions.len() {
            return Err("force array length does not match positions".into());
        }
        for bin in &mut self.bins {
            bin.clear();
        }
        for (i, &position) in positions.iter().enumerate() {
            if position.iter().any(|x| !x.is_finite()) {
                return Err("non-finite particle position".into());
            }
            let cell = self.index(position);
            self.bins[cell].push(i);
        }
        output.fill([0.0; 2]);
        let shift = crate::pair::energy(CUTOFF);
        let mut potential = 0.0;
        for i in 0..positions.len() {
            self.candidates.clear();
            for &cell in &self.neighbours[self.index(positions[i])] {
                for &j in &self.bins[cell] {
                    if j > i {
                        self.candidates.push(j);
                    }
                }
            }
            // Preserve naive i,j summation order while skipping distant cells.
            self.candidates.sort_unstable();
            for &j in &self.candidates {
                potential += add_pair(i, j, positions, self.box_size, shift, output)?;
            }
        }
        validate_force_result(potential, output)?;
        Ok(potential)
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
            potential += add_pair(i, j, positions, box_size, shift, output)?;
        }
    }
    validate_force_result(potential, output)?;
    Ok(potential)
}

#[inline(always)]
fn add_pair(
    i: usize,
    j: usize,
    positions: &[Vector],
    box_size: Vector,
    shift: f64,
    output: &mut [Vector],
) -> MdResult<f64> {
    let dx = minimum_image(positions[i][0] - positions[j][0], box_size[0]);
    let dy = minimum_image(positions[i][1] - positions[j][1], box_size[1]);
    let r2 = dx * dx + dy * dy;
    if !r2.is_finite() || r2 == 0.0 {
        return Err(format!("invalid minimum-image distance for pair {i},{j}"));
    }
    if r2 >= CUTOFF * CUTOFF {
        return Ok(0.0);
    }
    let r = r2.sqrt();
    let scale = crate::pair::force(r) / r;
    output[i][0] += scale * dx;
    output[i][1] += scale * dy;
    output[j][0] -= scale * dx;
    output[j][1] -= scale * dy;
    Ok(crate::pair::energy(r) - shift)
}

fn validate_force_result(potential: f64, output: &[Vector]) -> MdResult<()> {
    if !potential.is_finite() || output.iter().flatten().any(|x| !x.is_finite()) {
        return Err("non-finite periodic force or energy".into());
    }
    Ok(())
}
