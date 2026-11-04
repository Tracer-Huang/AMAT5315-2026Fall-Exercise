#[path = "bin/benchmark.rs"]
#[allow(dead_code)]
mod order;
use serde_json::json;
use std::{env, fs, path::Path};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;
#[derive(Clone)]
struct Factor {
    scope: Vec<usize>,
    vals: Vec<f64>,
}
fn read_model(path: &str) -> Result<(Vec<usize>, Vec<Factor>), String> {
    let raw = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut it = raw.split_whitespace();
    if it.next() != Some("MARKOV") {
        return Err("expected MARKOV".into());
    }
    fn num<T: std::str::FromStr>(i: &mut std::str::SplitWhitespace) -> Result<T, String> {
        i.next()
            .ok_or("unexpected EOF")?
            .parse()
            .map_err(|_| "invalid number".into())
    }
    let n: usize = num(&mut it)?;
    let mut dims = vec![];
    for _ in 0..n {
        dims.push(num(&mut it)?)
    }
    let nf: usize = num(&mut it)?;
    let mut scopes = vec![];
    for _ in 0..nf {
        let count: usize = num(&mut it)?;
        let mut s = vec![];
        for _ in 0..count {
            s.push(num(&mut it)?)
        }
        scopes.push(s)
    }
    let mut factors = vec![];
    for s in scopes {
        let count: usize = num(&mut it)?;
        if count != s.iter().map(|&x| dims[x]).product::<usize>() {
            return Err("wrong table size".into());
        }
        let mut vals = vec![];
        for _ in 0..count {
            vals.push(num(&mut it)?)
        }
        factors.push(Factor { scope: s, vals })
    }
    Ok((dims, factors))
}
fn enumerate(
    dims: &[usize],
    f: &[Factor],
    ev: &[(usize, usize)],
) -> (f64, Vec<Vec<f64>>, Vec<f64>) {
    let n = dims.len();
    let mut mar: Vec<Vec<f64>> = dims.iter().map(|d| vec![0.; *d]).collect();
    let mut joint = vec![0.; dims[1] * dims[3]];
    let mut z = 0.;
    for mut k in 0..dims.iter().product::<usize>() {
        let mut x = vec![0; n];
        for i in (0..n).rev() {
            x[i] = k % dims[i];
            k /= dims[i]
        }
        if ev.iter().any(|&(v, s)| x[v] != s) {
            continue;
        }
        let mut p = 1.;
        for factor in f {
            let idx = factor.scope.iter().fold(0, |acc, &v| acc * dims[v] + x[v]);
            p *= factor.vals[idx]
        }
        z += p;
        for i in 0..n {
            mar[i][x[i]] += p
        }
        joint[x[1] * dims[3] + x[3]] += p;
    }
    for a in &mut mar {
        for v in a {
            *v /= z
        }
    }
    for v in &mut joint {
        *v /= z
    }
    (z, mar, joint)
}
fn contract(
    dims: &[usize],
    f: &[Factor],
    ev: &[(usize, usize)],
    out: &[usize],
) -> Result<Vec<f64>, String> {
    let mut tensors = vec![];
    let mut terms = vec![];
    for factor in f {
        let scope: Vec<_> = factor.scope.iter().rev().copied().collect();
        let shape: Vec<_> = scope.iter().map(|&i| dims[i]).collect();
        tensors.push(
            Tensor::from_vec_col_major(shape, factor.vals.clone()).map_err(|e| e.to_string())?,
        );
        terms.push(
            scope
                .iter()
                .map(|&i| (b'a' + i as u8) as char)
                .collect::<String>(),
        );
    }
    for &(i, x) in ev {
        let mut v = vec![0.; dims[i]];
        v[x] = 1.;
        tensors.push(Tensor::from_vec_col_major([dims[i]], v).map_err(|e| e.to_string())?);
        terms.push(((b'a' + i as u8) as char).to_string())
    }
    // Reverse output so contiguous column-major equals UAI's last-index-fastest layout.
    let output: String = out
        .iter()
        .rev()
        .map(|&i| (b'a' + i as u8) as char)
        .collect();
    let equation = format!("{}->{}", terms.join(","), output);
    let refs: Vec<_> = tensors.iter().collect();
    let t = refs
        .as_slice()
        .einsum(&equation, &mut CpuEngine::serial())
        .map_err(|e| e.to_string())?;
    Ok(t.as_slice::<f64>().map_err(|e| e.to_string())?.to_vec())
}
fn norm(v: &mut [f64]) -> Result<(), String> {
    let z: f64 = v.iter().sum();
    if z <= 0. {
        return Err("zero message".into());
    }
    for x in v {
        *x /= z
    }
    Ok(())
}
fn bp(
    dims: &[usize],
    f: &[Factor],
    ev: &[(usize, usize)],
    exact: &[Vec<f64>],
    eta: f64,
    tol: f64,
    limit: usize,
) -> Result<serde_json::Value, String> {
    if !(0. < eta && eta <= 1.) || tol <= 0. {
        return Err("invalid BP settings".into());
    }
    let n = dims.len();
    let mut neigh = vec![vec![]; n];
    for (j, fac) in f.iter().enumerate() {
        for (a, &v) in fac.scope.iter().enumerate() {
            neigh[v].push((j, a));
        }
    }
    let mut vf: Vec<Vec<Vec<f64>>> = f
        .iter()
        .map(|fac| {
            fac.scope
                .iter()
                .map(|&v| vec![1. / dims[v] as f64; dims[v]])
                .collect()
        })
        .collect();
    let mut fv = vf.clone();
    let mut history = vec![];
    let mut work: usize = 0;
    for sweep in 0..=limit {
        let mut nv = vf.clone();
        let mut nf = fv.clone();
        let mut per = 0;
        for (j, fac) in f.iter().enumerate() {
            for (axis, &v) in fac.scope.iter().enumerate() {
                let mut msg = vec![1.; dims[v]];
                for &(g, a) in &neigh[v] {
                    if g != j {
                        for x in 0..dims[v] {
                            msg[x] *= fv[g][a][x];
                            per += 1
                        }
                    }
                }
                norm(&mut msg)?;
                nv[j][axis] = msg;
                let mut msg = vec![0.; dims[v]];
                for (idx, &weight) in fac.vals.iter().enumerate() {
                    let mut rem = idx;
                    let mut states = vec![0; fac.scope.len()];
                    for k in (0..fac.scope.len()).rev() {
                        let u = fac.scope[k];
                        states[k] = rem % dims[u];
                        rem /= dims[u]
                    }
                    if fac
                        .scope
                        .iter()
                        .enumerate()
                        .any(|(k, u)| ev.iter().any(|(e, x)| e == u && *x != states[k]))
                    {
                        continue;
                    }
                    let mut w = weight;
                    for (k, _) in fac.scope.iter().enumerate() {
                        if k != axis {
                            w *= vf[j][k][states[k]];
                            per += 1
                        }
                    }
                    msg[states[axis]] += w;
                    per += 1;
                }
                norm(&mut msg)?;
                nf[j][axis] = msg;
            }
        }
        let mut residual = 0f64;
        for (j, fac) in f.iter().enumerate() {
            for a in 0..fac.scope.len() {
                for x in 0..vf[j][a].len() {
                    residual = residual
                        .max((nv[j][a][x] - vf[j][a][x]).abs())
                        .max((nf[j][a][x] - fv[j][a][x]).abs())
                }
            }
        }
        let mut beliefs = vec![];
        let mut error = 0f64;
        for v in 0..n {
            let mut b = vec![1.; dims[v]];
            for &(j, a) in &neigh[v] {
                for x in 0..dims[v] {
                    b[x] *= fv[j][a][x]
                }
            }
            norm(&mut b)?;
            for (x, y) in b.iter().zip(&exact[v]) {
                error = error.max((x - y).abs())
            }
            beliefs.push(b)
        }
        history.push(json!({"sweep":sweep,"residual":residual,"cumulative_update_work":work,"beliefs":beliefs,"max_marginal_error":error}));
        if residual < tol || sweep == limit {
            return Ok(
                json!({"method":"bp","evidence":ev,"damping":eta,"tolerance":tol,"max_sweeps":limit,"converged":residual<tol,"sweeps":sweep,"residual":residual,"max_marginal_error":error,"marginals":beliefs,"history":history}),
            );
        }
        for (j, fac) in f.iter().enumerate() {
            for a in 0..fac.scope.len() {
                for x in 0..vf[j][a].len() {
                    vf[j][a][x] = (1. - eta) * vf[j][a][x] + eta * nv[j][a][x];
                    fv[j][a][x] = (1. - eta) * fv[j][a][x] + eta * nf[j][a][x];
                }
            }
        }
        work += per;
    }
    unreachable!()
}
fn run() -> Result<(), String> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args == ["--checks"] {
        let mut engine = CpuEngine::serial();
        let a =
            Tensor::from_vec_col_major([2, 2], vec![1., 3., 2., 4.]).map_err(|e| e.to_string())?;
        let b =
            Tensor::from_vec_col_major([2, 2], vec![2., 1., 0., 2.]).map_err(|e| e.to_string())?;
        let ab = [&a, &b]
            .einsum("ij,jk->ik", &mut engine)
            .map_err(|e| e.to_string())?;
        let trace = [&a, &b]
            .einsum("ij,ji->", &mut engine)
            .map_err(|e| e.to_string())?;
        let spin = [-1f64, 1.];
        let bond = Tensor::from_vec_col_major(
            [2, 2],
            spin.iter()
                .flat_map(|b| spin.iter().map(move |a| (0.5 * a * b).exp()))
                .collect(),
        )
        .map_err(|e| e.to_string())?;
        let z = [&bond, &bond, &bond]
            .einsum("ab,bc,ca->", &mut engine)
            .map_err(|e| e.to_string())?;
        println!(
            "{}",
            json!({"matrix_col_major":ab.as_slice::<f64>().unwrap(),"trace":trace.as_slice::<f64>().unwrap()[0],"ising":z.as_slice::<f64>().unwrap()[0]})
        );
        return Ok(());
    }
    if args.len() % 2 != 0 {
        return Err("expected flag/value pairs".into());
    }
    let opt = |key: &str| args.chunks(2).find(|x| x[0] == key).map(|x| x[1].as_str());
    for c in args.chunks(2) {
        if ![
            "--model",
            "--evidence",
            "--evidence-file",
            "--reference-mar",
            "--order",
            "--seed",
            "--plan",
            "--analyze",
            "--out",
            "--method",
            "--damping",
            "--tolerance",
            "--max-sweeps",
        ]
        .contains(&c[0].as_str())
        {
            return Err(format!("unsupported option {}", c[0]));
        }
    }
    let path = opt("--model").ok_or("missing --model")?;
    let evidence = opt("--evidence").unwrap_or("file");
    let dir = opt("--out").ok_or("missing --out")?;
    let ev = if let Some(file) = opt("--evidence-file") {
        let raw = fs::read_to_string(file).map_err(|e| e.to_string())?;
        let mut it = raw.split_whitespace();
        let count: usize = it
            .next()
            .ok_or("empty evidence")?
            .parse()
            .map_err(|_| "invalid evidence")?;
        let mut result = vec![];
        for _ in 0..count {
            let v = it
                .next()
                .ok_or("bad evidence")?
                .parse()
                .map_err(|_| "bad evidence")?;
            let x = it
                .next()
                .ok_or("bad evidence")?
                .parse()
                .map_err(|_| "bad evidence")?;
            result.push((v, x));
        }
        result
    } else {
        match evidence {
            "prior" => vec![],
            "AD" => vec![(0, 0), (7, 0)],
            "ADX" => vec![(0, 0), (7, 0), (6, 0)],
            _ => return Err("invalid evidence".into()),
        }
    };
    let (dims, f) = read_model(path)?;
    if ev.iter().any(|&(v, x)| v >= dims.len() || x >= dims[v]) {
        return Err("evidence out of range".into());
    }
    let reference = if let Some(file) = opt("--reference-mar") {
        let raw = fs::read_to_string(file).map_err(|e| e.to_string())?;
        let mut it = raw.split_whitespace();
        if it.next() != Some("MAR") {
            return Err("invalid MAR".into());
        }
        let n: usize = it.next().ok_or("bad MAR")?.parse().map_err(|_| "bad MAR")?;
        if n != dims.len() {
            return Err("MAR size mismatch".into());
        }
        let mut result = vec![];
        for &dim in &dims {
            let k: usize = it.next().ok_or("bad MAR")?.parse().map_err(|_| "bad MAR")?;
            if k != dim {
                return Err("MAR dimension mismatch".into());
            }
            let mut x = vec![];
            for _ in 0..k {
                x.push(
                    it.next()
                        .ok_or("bad MAR")?
                        .parse::<f64>()
                        .map_err(|_| "bad MAR")?,
                )
            }
            result.push(x)
        }
        Some(result)
    } else {
        None
    };
    let small = dims.len() == 8 && dims.iter().all(|d| *d == 2);
    let enumerated = if small && opt("--analyze").is_none() {
        Some(enumerate(&dims, &f, &ev))
    } else {
        None
    };
    let method = opt("--method").unwrap_or("exact");
    if method == "exact"
        && ["--damping", "--tolerance", "--max-sweeps"]
            .iter()
            .any(|x| opt(x).is_some())
    {
        return Err("BP options require --method bp".into());
    }
    if method == "bp"
        && ["--order", "--seed", "--plan", "--analyze"]
            .iter()
            .any(|x| opt(x).is_some())
    {
        return Err("ordering options are not applicable to BP".into());
    }
    if method == "bp" {
        let eta: f64 = opt("--damping")
            .ok_or("missing --damping")?
            .parse()
            .map_err(|_| "invalid damping")?;
        let tol: f64 = opt("--tolerance")
            .ok_or("missing --tolerance")?
            .parse()
            .map_err(|_| "invalid tolerance")?;
        let limit: usize = opt("--max-sweeps")
            .ok_or("missing --max-sweeps")?
            .parse()
            .map_err(|_| "invalid limit")?;
        let mar = reference
            .as_ref()
            .or_else(|| enumerated.as_ref().map(|x| &x.1))
            .ok_or("BP requires --reference-mar for this model")?;
        let report = bp(&dims, &f, &ev, mar, eta, tol, limit)?;
        fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        fs::write(
            Path::new(dir).join("run.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .map_err(|e| e.to_string())?;
        for x in report["history"].as_array().unwrap() {
            println!(
                "{} {} {}",
                x["sweep"], x["residual"], x["cumulative_update_work"]
            )
        }
        return Ok(());
    } else if opt("--method").unwrap_or("exact") != "exact" {
        return Err("invalid method".into());
    }
    let modes = [
        opt("--order").is_some(),
        opt("--plan").is_some(),
        opt("--analyze").is_some(),
    ];
    if modes.iter().filter(|x| **x).count() > 1 {
        return Err("order, plan and analyze are mutually exclusive".into());
    }
    let choice = if opt("--analyze").is_some() {
        "analyze"
    } else {
        opt("--order").unwrap_or("greedy")
    };
    let seed = opt("--seed")
        .map(|s| s.parse::<u64>().map_err(|_| "invalid seed".to_string()))
        .transpose()?;
    if seed.is_some() && choice != "random" {
        return Err("seed allowed only for random order".into());
    }
    let planned = order::run_order(
        path,
        &ev,
        choice,
        seed,
        opt("--plan").or_else(|| opt("--analyze")),
        dir,
    )?;
    if choice == "analyze" {
        return Ok(());
    }
    let (z, mar, joint) = enumerated.ok_or("exact infer supports binary ASIA only")?;
    let planned_mass = planned.ok_or("missing planned result")?;
    if ((planned_mass - z) / z).abs() >= 1e-10 {
        return Err("planned contraction differs from enumeration".into());
    }

    let z2 = contract(&dims, &f, &ev, &[])?[0];
    let mut cm = vec![];
    let mut err = 0f64;
    for i in 0..8 {
        let v: Vec<_> = contract(&dims, &f, &ev, &[i])?
            .iter()
            .map(|x| x / z2)
            .collect();
        for (a, b) in v.iter().zip(&mar[i]) {
            err = err.max((a - b).abs())
        }
        cm.push(v)
    }
    let cj: Vec<_> = contract(&dims, &f, &ev, &[1, 3])?
        .iter()
        .map(|x| x / z2)
        .collect();
    let je = cj
        .iter()
        .zip(&joint)
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    let report = json!({"evidence":evidence,"evidence_mass":z,"contraction_mass":z2,"planned_mass":planned_mass,"relative_mass_error":((z-z2)/z).abs(),"marginals":mar,"contraction_marginals":cm,"joint_T_L":joint,"contraction_joint_T_L":cj,"max_marginal_error":err,"max_joint_error":je});
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    fs::write(
        Path::new(dir).join("run.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
