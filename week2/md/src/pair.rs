//! Plain Lennard-Jones pair potential in reduced units.

pub fn energy(_r: f64) -> f64 {
    todo!("Part 2: implement pair energy")
}

/// Positive means repulsion; multiply by (x_i - x_j) / r for force on i.
pub fn force(_r: f64) -> f64 {
    todo!("Part 2: implement analytic radial force")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn well_depth() {
        let r0 = 2.0_f64.powf(1.0 / 6.0);
        assert!((energy(r0) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn force_is_minus_energy_derivative() {
        let h = 1e-5;
        for r in [0.95, 1.0, 1.08, 2.0_f64.powf(1.0 / 6.0), 1.2, 1.5, 2.4] {
            let numerical = -(energy(r + h) - energy(r - h)) / (2.0 * h);
            let analytical = force(r);
            let tolerance = 1e-6 * analytical.abs().max(1.0);
            assert!(
                (numerical - analytical).abs() < tolerance,
                "r={r}: analytic={analytical}, numerical={numerical}, tolerance={tolerance}"
            );
        }
    }
}
