use crate::{Args, Mode, Order};
use num_bigint::BigUint;
use omeco::{
    EinCode, GreedyMethod, NestedEinsum, ScoreFunction, TreeSA, contraction_complexity,
    optimize_code,
};
use rand::{Rng, rngs::StdRng};
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, HashMap},
    fs,
};

struct Factor {
    scope: Vec<usize>,
    energy: Vec<i64>,
}
fn pair(i: usize, j: usize, coupling: i64) -> Factor {
    Factor {
        scope: vec![i, j],
        energy: vec![coupling, -coupling, -coupling, coupling],
    }
}

fn network(
    args: &Args,
    rng: &mut StdRng,
) -> Result<(usize, Vec<Factor>), Box<dyn std::error::Error>> {
    let mut factors = Vec::new();
    let n = if let Some(path) = &args.input {
        let bundle: Value = serde_json::from_slice(&fs::read(path)?)?;
        let target = &bundle["target"];
        if target["type"] != "SpinGlass" {
            return Err("expected a SpinGlass bundle".into());
        }
        let data = &target["data"];
        let edges: Vec<[usize; 2]> = serde_json::from_value(data["graph"]["edges"].clone())?;
        let couplings: Vec<i64> = serde_json::from_value(data["couplings"].clone())?;
        let fields: Vec<i64> = serde_json::from_value(data["fields"].clone())?;
        let n = fields.len();
        if n == 0
            || edges.len() != couplings.len()
            || edges.iter().any(|&[i, j]| i >= n || j >= n || i == j)
        {
            return Err("invalid SpinGlass dimensions".into());
        }
        factors.extend(
            edges
                .into_iter()
                .zip(couplings)
                .filter(|(_, j)| *j != 0)
                .map(|([i, j], c)| pair(i, j, c)),
        );
        let connected: BTreeSet<_> = factors
            .iter()
            .flat_map(|f| f.scope.iter().copied())
            .collect();
        for (i, h) in fields.into_iter().enumerate() {
            if h != 0 || !connected.contains(&i) {
                factors.push(Factor {
                    scope: vec![i],
                    energy: vec![-h, h],
                });
            }
        }
        n
    } else {
        let l = args.l.ok_or("provide --l, --input or --n")?;
        let p = args.p_af.ok_or("lattice requires --p-af")?;
        if l == 0 || !(0.0..=1.0).contains(&p) {
            return Err("invalid lattice parameters".into());
        }
        let n = l.checked_mul(l).ok_or("lattice too large")?;
        for i in 0..n {
            if i % l + 1 < l {
                factors.push(pair(i, i + 1, if rng.random_bool(p) { 1 } else { -1 }));
            }
            if i + l < n {
                factors.push(pair(i, i + l, if rng.random_bool(p) { 1 } else { -1 }));
            }
        }
        if n == 1 {
            factors.push(Factor {
                scope: vec![0],
                energy: vec![0, 0],
            });
        }
        n
    };
    Ok((n, factors))
}

struct Cell {
    weight: f64,
    energy: i64,
    degeneracy: BigUint,
}
struct Tensor {
    cells: Vec<Cell>,
    log_scale: f64,
    branch: Option<Branch>,
}
struct Branch {
    children: Vec<Tensor>,
    projections: Vec<Vec<usize>>,
    output: Vec<usize>,
    summed: Vec<usize>,
}
fn mask(bits: &[usize], state: usize) -> usize {
    bits.iter()
        .enumerate()
        .map(|(i, b)| ((state >> i) & 1) << b)
        .sum()
}
fn project(state: usize, positions: &[usize]) -> usize {
    positions
        .iter()
        .enumerate()
        .map(|(i, p)| ((state >> p) & 1) << i)
        .sum()
}
impl Branch {
    fn terms(&self, state: usize) -> impl Iterator<Item = &Cell> {
        self.children
            .iter()
            .zip(&self.projections)
            .map(move |(child, positions)| &child.cells[project(state, positions)])
    }
    fn weight(&self, state: usize) -> f64 {
        self.terms(state).map(|cell| cell.weight).product()
    }
}
fn contract(tree: &NestedEinsum<usize>, factors: &[Factor], beta: f64) -> Tensor {
    match tree {
        NestedEinsum::Leaf { tensor_index } => {
            let factor = &factors[*tensor_index];
            let minimum = *factor.energy.iter().min().unwrap();
            Tensor {
                cells: factor
                    .energy
                    .iter()
                    .map(|&energy| Cell {
                        weight: (-beta * (energy - minimum) as f64).exp(),
                        energy,
                        degeneracy: 1u32.into(),
                    })
                    .collect(),
                log_scale: -beta * minimum as f64,
                branch: None,
            }
        }
        NestedEinsum::Node { args, eins } => {
            let children: Vec<_> = args
                .iter()
                .map(|arg| contract(arg, factors, beta))
                .collect();
            let vars: Vec<_> = eins
                .ixs
                .iter()
                .flatten()
                .chain(&eins.iy)
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            let positions = |indices: &[usize]| {
                indices
                    .iter()
                    .map(|i| vars.binary_search(i).unwrap())
                    .collect::<Vec<_>>()
            };
            let branch = Branch {
                children,
                projections: eins.ixs.iter().map(|v| positions(v)).collect(),
                output: positions(&eins.iy),
                summed: positions(
                    &vars
                        .iter()
                        .copied()
                        .filter(|i| !eins.iy.contains(i))
                        .collect::<Vec<_>>(),
                ),
            };
            let mut cells = (0..1 << eins.iy.len())
                .map(|_| Cell {
                    weight: 0.,
                    energy: i64::MAX,
                    degeneracy: 0u32.into(),
                })
                .collect::<Vec<_>>();
            for state in 0..1 << vars.len() {
                let cell = &mut cells[project(state, &branch.output)];
                cell.weight += branch.weight(state);
                let energy = branch.terms(state).map(|c| c.energy).sum();
                if energy <= cell.energy {
                    let count: BigUint =
                        branch.terms(state).map(|c| c.degeneracy.clone()).product();
                    if energy == cell.energy {
                        cell.degeneracy += count;
                    } else {
                        cell.energy = energy;
                        cell.degeneracy = count;
                    }
                }
            }
            let scale = cells.iter().map(|c| c.weight).fold(0., f64::max);
            for cell in &mut cells {
                cell.weight /= scale;
            }
            let log_scale = scale.ln() + branch.children.iter().map(|c| c.log_scale).sum::<f64>();
            Tensor {
                cells,
                log_scale,
                branch: Some(branch),
            }
        }
    }
}

fn draw(tree: &NestedEinsum<usize>, tensor: &Tensor, bits: &mut [usize], rng: &mut StdRng) {
    let NestedEinsum::Node { args, eins } = tree else {
        return;
    };
    let branch = tensor.branch.as_ref().unwrap();
    let vars: Vec<_> = eins
        .ixs
        .iter()
        .flatten()
        .chain(&eins.iy)
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let fixed = eins
        .iy
        .iter()
        .enumerate()
        .map(|(i, &v)| bits[v] << branch.output[i])
        .sum::<usize>();
    let states = || (0..1 << branch.summed.len()).map(|s| fixed | mask(&branch.summed, s));
    let total = states().map(|s| branch.weight(s)).sum::<f64>();
    let mut threshold = rng.random::<f64>() * total;
    let mut selected = fixed;
    for state in states() {
        let weight = branch.weight(state);
        if weight > 0. {
            selected = state;
        }
        if threshold < weight {
            break;
        }
        threshold -= weight;
    }
    for &i in &branch.summed {
        bits[vars[i]] = (selected >> i) & 1;
    }
    for (child, subtree) in branch.children.iter().zip(args) {
        draw(subtree, child, bits, rng);
    }
}

pub fn run(args: &Args, rng: &mut StdRng) -> Result<(Value, Value), Box<dyn std::error::Error>> {
    let (n, factors) = network(args, rng)?;
    let code = EinCode::new(factors.iter().map(|f| f.scope.clone()).collect(), vec![]);
    let dims: HashMap<_, _> = (0..n).map(|i| (i, 2)).collect();
    let tree = if factors.len() == 1 {
        NestedEinsum::node(vec![NestedEinsum::leaf(0)], code.clone())
    } else {
        match args.order {
            Order::Greedy => optimize_code(&code, &dims, &GreedyMethod::default()),
            Order::Treesa => {
                let mut planner = TreeSA::fast();
                planner.score = ScoreFunction::time_optimized();
                optimize_code(&code, &dims, &planner)
            }
        }
        .ok_or("planning failed")?
    };
    let c = contraction_complexity(&tree, &dims, &code.ixs);
    let mut summary = json!({"variables":n,"sc":c.sc,"tc":c.tc});
    let mut evidence = json!({"tree":tree,"factors":factors.iter().map(|f|json!({"scope":f.scope,"energy":f.energy})).collect::<Vec<_>>()});
    if matches!(args.mode, Mode::Plan) {
        return Ok((summary, evidence));
    }
    let beta = args
        .beta
        .filter(|b| b.is_finite() && *b >= 0.)
        .ok_or("provide finite nonnegative --beta")?;
    if c.sc > 23. || c.tc > 29. {
        return Err("planned contraction exceeds local exact-evaluation budget".into());
    }
    let tensor = contract(&tree, &factors, beta);
    if tensor.cells.len() != 1 {
        return Err("expected scalar contraction".into());
    }
    let cell = &tensor.cells[0];
    let log_z = tensor.log_scale + cell.weight.ln();
    if !log_z.is_finite() {
        return Err("nonfinite partition function".into());
    }
    summary["log_z"] = json!(log_z);
    summary["e_min"] = json!(cell.energy);
    summary["g0"] = json!(cell.degeneracy.to_string());
    if matches!(args.mode, Mode::Sample) {
        let mut samples = Vec::new();
        for _ in 0..args.samples.unwrap() {
            let mut bits = vec![0; n];
            draw(&tree, &tensor, &mut bits, rng);
            let energy: i64 = factors
                .iter()
                .map(|f| {
                    let index = f
                        .scope
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| bits[v] << i)
                        .sum::<usize>();
                    f.energy[index]
                })
                .sum();
            samples.push(json!({"bits":bits,"energy":energy}));
        }
        evidence["samples"] = json!(samples);
    }
    Ok((summary, evidence))
}
