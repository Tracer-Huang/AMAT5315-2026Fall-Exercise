use week4_fluid::{flow::*, io::*};
fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let case = args
        .next()
        .ok_or("usage: field {taylor-green|random} --n N [--seed S --k-min A --k-max B]")?;
    if case == "--help" {
        println!("field taylor-green --n N\nfield random --n N --seed S --k-min A --k-max B");
        return Ok(());
    }
    let allowed = match case.as_str() {
        "taylor-green" => vec!["--n"],
        "random" => vec!["--n", "--seed", "--k-min", "--k-max"],
        _ => return Err("unknown field case".into()),
    };
    let map = flags(args, &allowed)?;
    let n = required(&map, "--n")?;
    valid_n(n)?;
    let (u, v, seed, k_band) = if case == "random" {
        let seed = required(&map, "--seed")?;
        let lo: usize = required(&map, "--k-min")?;
        let hi: usize = required(&map, "--k-max")?;
        if lo < 1 || lo > hi || hi > n / 3 {
            return Err("require 1 <= k-min <= k-max <= floor(n/3)".into());
        }
        let (u, v) = random_field(n, seed, lo, hi);
        (u, v, Some(seed), Some([lo, hi]))
    } else {
        let (u, v) = taylor_green(n);
        (u, v, None, None)
    };
    serde_json::to_writer(
        std::io::stdout().lock(),
        &Initial {
            case,
            n,
            seed,
            k_band,
            u,
            v,
        },
    )?;
    println!();
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("field: {e}");
        std::process::exit(1);
    }
}
