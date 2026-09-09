//! Export the actual library energy and force for the Part 2 plot.
use std::io::{self, BufWriter, Write};

fn main() -> io::Result<()> {
    let mut out = BufWriter::new(io::stdout().lock());
    writeln!(out, "x,y,energy,fx,fy")?;
    for iy in 0..=120 {
        for ix in 0..=120 {
            let x = -2.5 + 5.0 * ix as f64 / 120.0;
            let y = -2.5 + 5.0 * iy as f64 / 120.0;
            let r = x.hypot(y);
            if r == 0.0 {
                writeln!(out, "{x},{y},NaN,0,0")?;
            } else {
                let radial = md::pair::force(r);
                writeln!(
                    out,
                    "{x},{y},{},{},{}",
                    md::pair::energy(r),
                    radial * x / r,
                    radial * y / r
                )?;
            }
        }
    }
    Ok(())
}
