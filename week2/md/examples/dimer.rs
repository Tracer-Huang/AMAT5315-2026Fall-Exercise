use md::dynamics::{ForwardEuler, Integrator, MdResult, System, VelocityVerlet, advance};
use std::io::{self, BufWriter, Write};

fn measure(method: &impl Integrator, steps: usize) -> MdResult<Vec<f64>> {
    let mut system = System::new(vec![[0.0, 0.0], [1.2, 0.0]], vec![[0.0; 2]; 2])?;
    let initial = system.total_energy();
    let mut errors = vec![0.0];
    for _ in 0..steps {
        advance(method, &mut system, 0.01)?;
        errors.push((system.total_energy() - initial) / initial.abs());
    }
    Ok(errors)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let euler = measure(&ForwardEuler, 500)?;
    let verlet = measure(&VelocityVerlet, 5000)?;
    let mut out = BufWriter::new(io::stdout().lock());
    writeln!(out, "time,euler,verlet")?;
    for (step, error) in verlet.iter().enumerate() {
        writeln!(
            out,
            "{},{},{}",
            step as f64 * 0.01,
            euler.get(step).copied().unwrap_or(f64::NAN),
            error
        )?;
    }
    Ok(())
}
