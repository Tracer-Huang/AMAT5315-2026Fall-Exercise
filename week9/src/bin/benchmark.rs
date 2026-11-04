use omeco::{
    contraction_complexity, optimize_code, EinCode, GreedyMethod, NestedEinsum, ScoreFunction,
    TreeSA,
};
use serde_json::json;
use std::{
    collections::{BTreeSet, HashMap},
    env, fs,
    path::Path,
    time::Instant,
};
use tenferro_cpu::CpuEngine;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;
#[derive(Clone)]
struct F {
    scope: Vec<usize>,
    v: Vec<f64>,
}
fn read(p: &str) -> (Vec<usize>, Vec<F>) {
    let s = fs::read_to_string(p).unwrap();
    let mut t = s.split_whitespace();
    assert_eq!(t.next(), Some("MARKOV"));
    let n: usize = t.next().unwrap().parse().unwrap();
    let d: Vec<usize> = (0..n).map(|_| t.next().unwrap().parse().unwrap()).collect();
    let nf: usize = t.next().unwrap().parse().unwrap();
    let scopes: Vec<Vec<usize>> = (0..nf)
        .map(|_| {
            let n: usize = t.next().unwrap().parse().unwrap();
            (0..n).map(|_| t.next().unwrap().parse().unwrap()).collect()
        })
        .collect();
    let f = scopes
        .into_iter()
        .map(|scope| {
            let n: usize = t.next().unwrap().parse().unwrap();
            assert_eq!(n, scope.iter().map(|&i| d[i]).product::<usize>());
            let v = (0..n).map(|_| t.next().unwrap().parse().unwrap()).collect();
            F { scope, v }
        })
        .collect();
    (d, f)
}
fn condition(f: &[F], dims: &[usize], ev: &HashMap<usize, usize>) -> Vec<F> {
    f.iter()
        .map(|f| {
            let scope: Vec<_> = f
                .scope
                .iter()
                .copied()
                .filter(|v| !ev.contains_key(v))
                .collect();
            let n = scope.iter().map(|&i| dims[i]).product::<usize>();
            let mut vals = vec![];
            for mut x in 0..n {
                let mut states = ev.clone();
                for &i in scope.iter().rev() {
                    states.insert(i, x % dims[i]);
                    x /= dims[i]
                }
                let idx = f.scope.iter().fold(0, |acc, i| acc * dims[*i] + states[i]);
                vals.push(f.v[idx])
            }
            F { scope, v: vals }
        })
        .collect()
}
struct Scaled {
    t: Tensor,
    log: f64,
}
fn build(f: &F, dims: &[usize]) -> Scaled {
    let scale = f.v.iter().copied().fold(0.0_f64, f64::max);
    assert!(scale > 0.);
    let shape: Vec<_> = f.scope.iter().rev().map(|&i| dims[i]).collect();
    Scaled {
        t: Tensor::from_vec_col_major(shape, f.v.iter().map(|x| x / scale).collect()).unwrap(),
        log: scale.ln(),
    }
}
fn eval(tree: &NestedEinsum<usize>, fs: &[F], dims: &[usize], engine: &mut CpuEngine) -> Scaled {
    match tree {
        NestedEinsum::Leaf { tensor_index } => build(&fs[*tensor_index], dims),
        NestedEinsum::Node { args, eins } => {
            let children: Vec<_> = args.iter().map(|x| eval(x, fs, dims, engine)).collect();
            let all: BTreeSet<_> = eins.ixs.iter().flatten().copied().collect();
            assert!(all.len() <= 52);
            let labels: HashMap<_, _> = all
                .into_iter()
                .enumerate()
                .map(|(i, v)| {
                    (
                        v,
                        if i < 26 {
                            (b'a' + i as u8) as char
                        } else {
                            (b'A' + (i - 26) as u8) as char
                        },
                    )
                })
                .collect();
            let term = |v: &Vec<usize>| v.iter().rev().map(|x| labels[x]).collect::<String>();
            let equation = format!(
                "{}->{}",
                eins.ixs.iter().map(term).collect::<Vec<_>>().join(","),
                term(&eins.iy)
            );
            let refs: Vec<_> = children.iter().map(|x| &x.t).collect();
            let tensor = refs
                .as_slice()
                .einsum(&equation, engine)
                .unwrap_or_else(|e| panic!("{equation}: {e}"));
            let data = tensor.as_slice::<f64>().unwrap();
            let scale = data.iter().copied().fold(0.0_f64, f64::max);
            assert!(scale > 0.);
            let shape: Vec<_> = eins.iy.iter().rev().map(|&i| dims[i]).collect();
            Scaled {
                t: Tensor::from_vec_col_major(shape, data.iter().map(|x| x / scale).collect())
                    .unwrap(),
                log: children.iter().map(|x| x.log).sum::<f64>() + scale.ln(),
            }
        }
    }
}
fn random_plan(code: &EinCode<usize>, mut seed: u64) -> NestedEinsum<usize> {
    let mut items: Vec<_> = code
        .ixs
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, s)| (NestedEinsum::leaf(i), s))
        .collect();
    while items.len() > 1 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let a = (seed as usize) % items.len();
        let first = items.remove(a);
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let b = (seed as usize) % items.len();
        let second = items.remove(b);
        let union: BTreeSet<_> = first.1.iter().chain(&second.1).copied().collect();
        let outside: BTreeSet<_> = items
            .iter()
            .flat_map(|x| x.1.iter())
            .chain(code.iy.iter())
            .copied()
            .collect();
        let output: Vec<_> = union.intersection(&outside).copied().collect();
        let eins = EinCode::new(vec![first.1, second.1], output.clone());
        items.push((NestedEinsum::node(vec![first.0, second.0], eins), output));
    }
    items.pop().unwrap().0
}
fn main() {
    let args: Vec<_> = env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "benchmark MODEL EVIDENCE (0=0,7=0) greedy|treesa OUT"
    );
    let start = Instant::now();
    let (dims, f) = read(&args[1]);
    let ev: HashMap<usize, usize> = args[2]
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            let (a, b) = s.split_once('=').unwrap();
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .collect();
    let factors = condition(&f, &dims, &ev);
    let sizes: HashMap<_, _> = dims.iter().copied().enumerate().collect();
    let method = &args[3];
    let queries: Vec<Vec<usize>> =
        if dims.len() > 100 && !method.starts_with("replay:") && !method.starts_with("analyze:") {
            vec![vec![], vec![59], vec![102], vec![139]]
        } else {
            vec![vec![]]
        };
    let dest = Path::new(&args[4]);
    fs::create_dir_all(dest).unwrap();
    let mut reports = vec![];
    for out in queries {
        let code = EinCode::new(
            factors.iter().map(|f| f.scope.clone()).collect(),
            out.clone(),
        );
        let now = Instant::now();
        let tree = if method == "greedy" {
            optimize_code(&code, &sizes, &GreedyMethod::default()).unwrap()
        } else if method == "treesa" {
            let mut sa = TreeSA::fast();
            sa.score = ScoreFunction::time_optimized();
            optimize_code(&code, &sizes, &sa).unwrap()
        } else if let Some(seed) = method.strip_prefix("random:") {
            random_plan(&code, seed.parse().expect("invalid random seed"))
        } else if let Some(path) = method
            .strip_prefix("replay:")
            .or_else(|| method.strip_prefix("analyze:"))
        {
            serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
        } else {
            panic!("unknown method")
        };
        let planning = now.elapsed().as_secs_f64();
        let c = contraction_complexity(&tree, &sizes, &code.ixs);
        let treefile = if out.is_empty() {
            "plan-z.json".to_string()
        } else {
            format!("plan-{}.json", out[0])
        };
        fs::write(dest.join(treefile), serde_json::to_vec(&tree).unwrap()).unwrap();
        if method.starts_with("analyze:") {
            reports.push(
                json!({"query":out,"work_log2":c.tc,"space_log2":c.sc,"planning_seconds":planning}),
            );
            continue;
        }
        let t = Instant::now();
        let res = eval(&tree, &factors, &dims, &mut CpuEngine::serial());
        let vals = res.t.as_slice::<f64>().unwrap();
        let logz = (res.log + vals.iter().sum::<f64>().ln()) / std::f64::consts::LN_10;
        let probs: Vec<_> = vals.iter().map(|x| x / vals.iter().sum::<f64>()).collect();
        reports.push(json!({"query":out,"log10_z":logz,"probabilities":probs,"work_log2":c.tc,"space_log2":c.sc,"planning_seconds":planning,"execution_seconds":t.elapsed().as_secs_f64()}));
        println!("{}", reports.last().unwrap());
    }
    fs::write(dest.join("result.json"),serde_json::to_vec_pretty(&json!({"method":method,"log10_z":reports[0]["log10_z"],"queries":reports,"elapsed_seconds":start.elapsed().as_secs_f64(),"treesa":"fast, 1 trial, seed 42, time-only"})).unwrap()).unwrap();
}

/// PDF-compatible ASIA scalar planning, analysis and replay. Plans bind the input
/// model byte-for-byte and evidence; tree structure and retained indices are checked.
pub fn run_order(
    model: &str,
    evidence: &[(usize, usize)],
    choice: &str,
    seed: Option<u64>,
    saved: Option<&str>,
    out: &str,
) -> Result<Option<f64>, String> {
    let model_text = fs::read_to_string(model).map_err(|e| e.to_string())?;
    let (dims, raw) = read(model);
    let ev: HashMap<usize, usize> = evidence.iter().copied().collect();
    let factors = condition(&raw, &dims, &ev);
    let code = EinCode::new(factors.iter().map(|f| f.scope.clone()).collect(), vec![]);
    let sizes: HashMap<_, _> = dims.iter().copied().enumerate().collect();
    let (tree, method, used_seed): (NestedEinsum<usize>, String, Option<u64>) = if let Some(path) =
        saved
    {
        let doc: serde_json::Value =
            serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if doc["model"] != model_text || doc["evidence"] != json!(evidence) {
            return Err("plan model or evidence does not match".into());
        }
        (
            serde_json::from_value(doc["tree"].clone()).map_err(|e| e.to_string())?,
            doc["method"].as_str().ok_or("missing method")?.to_string(),
            doc["seed"].as_u64(),
        )
    } else {
        match choice {
            "greedy" => (
                optimize_code(&code, &sizes, &GreedyMethod::default()).ok_or("planning failed")?,
                "greedy".into(),
                None,
            ),
            "random" => {
                let seed = seed.ok_or("random requires seed")?;
                (random_plan(&code, seed), "random".into(), Some(seed))
            }
            _ => return Err("invalid order".into()),
        }
    };
    let mut total = HashMap::new();
    for f in &factors {
        for &v in &f.scope {
            *total.entry(v).or_insert(0usize) += 1
        }
    }
    fn walk(
        t: &NestedEinsum<usize>,
        f: &[F],
        dims: &[usize],
        total: &HashMap<usize, usize>,
        seen: &mut BTreeSet<usize>,
        steps: &mut Vec<serde_json::Value>,
    ) -> Result<(Vec<usize>, HashMap<usize, usize>), String> {
        match t {
            NestedEinsum::Leaf { tensor_index: i } => {
                if *i >= f.len() || !seen.insert(*i) {
                    return Err("invalid plan leaves".into());
                }
                let s = f[*i].scope.clone();
                let mut counts = HashMap::new();
                for &v in &s {
                    *counts.entry(v).or_insert(0) += 1
                }
                Ok((s, counts))
            }
            NestedEinsum::Node { args, eins } => {
                if args.len() != 2 || eins.ixs.len() != 2 {
                    return Err("plan must be binary".into());
                }
                let (s1, mut c1) = walk(&args[0], f, dims, total, seen, steps)?;
                let (s2, c2) = walk(&args[1], f, dims, total, seen, steps)?;
                if eins.ixs != vec![s1.clone(), s2.clone()] {
                    return Err("plan input labels mismatch".into());
                }
                for (k, v) in c2 {
                    *c1.entry(k).or_insert(0) += v
                }
                let union: BTreeSet<_> = s1.iter().chain(&s2).copied().collect();
                let expected: BTreeSet<_> = union
                    .iter()
                    .copied()
                    .filter(|v| c1.get(v).unwrap_or(&0) < total.get(v).unwrap_or(&0))
                    .collect();
                let actual: BTreeSet<_> = eins.iy.iter().copied().collect();
                if !expected.is_subset(&actual)
                    || !actual.is_subset(&union)
                    || eins.iy.len() != actual.len()
                {
                    return Err("invalid retained labels".into());
                }
                let work = union.iter().map(|&v| dims[v]).product::<usize>();
                let entries = eins.iy.iter().map(|&v| dims[v]).product::<usize>();
                steps.push(json!({"inputs":[s1,s2],"union":union,"retained":eins.iy,"work":work,"entries":entries}));
                Ok((eins.iy.clone(), c1))
            }
        }
    }
    let mut seen = BTreeSet::new();
    let mut steps = vec![];
    let (scope, _) = walk(&tree, &factors, &dims, &total, &mut seen, &mut steps)?;
    if !scope.is_empty() || seen.len() != factors.len() {
        return Err("incomplete scalar plan".into());
    }
    let work: usize = steps
        .iter()
        .map(|x| x["work"].as_u64().unwrap() as usize)
        .sum();
    let largest = factors
        .iter()
        .map(|f| f.v.len())
        .chain(
            steps
                .iter()
                .map(|x| x["entries"].as_u64().unwrap() as usize),
        )
        .max()
        .unwrap_or(1);
    fs::create_dir_all(out).map_err(|e| e.to_string())?;
    let order = json!({"model":model_text,"evidence":evidence,"method":method,"seed":used_seed,"tree":tree});
    fs::write(
        Path::new(out).join("order.json"),
        serde_json::to_vec_pretty(&order).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    let complexity = json!({"work":work,"largest_tensor":largest,"steps":steps});
    fs::write(
        Path::new(out).join("complexity.json"),
        serde_json::to_vec_pretty(&complexity).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    if choice == "analyze" {
        return Ok(None);
    }
    let value = eval(&tree, &factors, &dims, &mut CpuEngine::serial());
    let scalar = value.t.as_slice::<f64>().map_err(|e| e.to_string())?[0] * value.log.exp();
    Ok(Some(scalar))
}
