use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, io::Write, str::FromStr};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Initial {
    pub case: String,
    pub n: usize,
    pub seed: Option<u64>,
    pub k_band: Option<[usize; 2]>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}
pub fn flags(
    args: impl Iterator<Item = String>,
    allowed: &[&str],
) -> Result<BTreeMap<String, String>> {
    let mut iter = args;
    let mut map = BTreeMap::new();
    while let Some(key) = iter.next() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unknown argument: {key}").into());
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        if map.insert(key.clone(), value).is_some() {
            return Err(format!("duplicate argument: {key}").into());
        }
    }
    Ok(map)
}
pub fn required<T: FromStr>(map: &BTreeMap<String, String>, key: &str) -> Result<T> {
    map.get(key)
        .ok_or_else(|| format!("required argument: {key}"))?
        .parse()
        .map_err(|_| format!("invalid value for {key}").into())
}
pub fn optional<T: FromStr>(map: &BTreeMap<String, String>, key: &str, default: T) -> Result<T> {
    if map.contains_key(key) {
        required(map, key)
    } else {
        Ok(default)
    }
}
pub fn valid_n(n: usize) -> Result<()> {
    if n < 4 || !n.is_power_of_two() {
        return Err("n must be a power of two >= 4".into());
    }
    Ok(())
}
pub fn positive(x: f64, name: &str, zero: bool) -> Result<()> {
    if !x.is_finite() || x < 0. || (!zero && x == 0.) {
        return Err(format!(
            "{name} must be finite and {}",
            if zero { "nonnegative" } else { "positive" }
        )
        .into());
    }
    Ok(())
}
pub fn array(writer: &mut impl Write, values: impl Iterator<Item = f64>) -> Result<()> {
    write!(writer, "[")?;
    for (i, x) in values.enumerate() {
        if !x.is_finite() {
            return Err("nonfinite simulation output; reduce dt".into());
        }
        if i > 0 {
            write!(writer, ",")?;
        }
        write!(writer, "{x:.6}")?;
    }
    write!(writer, "]")?;
    Ok(())
}
