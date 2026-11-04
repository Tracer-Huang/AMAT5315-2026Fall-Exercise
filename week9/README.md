# Week 9: probabilistic inference by tensor contraction

Local review package; not submitted. CPU f64 implementation in Rust 1.98.1 with tenferro 0.9.0 and omeco 0.2.6. `Cargo.lock` pins dependencies; Rust 1.98.1 successfully compiled and tested them. This is newer than the sheet's 1.96.0 and required no toolchain downgrade.

## Evidence and conclusions

- [Study notes in Chinese](STUDY_NOTES.zh.md). [AD query factor graph](evidence/graphs.svg); [disease probabilities](evidence/posteriors.png); [joint vs product](evidence/joint.png); [BP residual and error](evidence/bp-traces.png).
- ASIA exact enumerates and contracts the original eight UAI tables. `artifacts/{prior,AD,ADX}/run.json` retains both answers, all node marginals, T–L joint and errors. Evidence masses are 1, 0.004501375, 0.00098822675. The matrix and Ising checks are Rust integration tests.
- [Saved contraction trees](evidence/orders.png): random seed 42 has work 96 and largest tensor 16; omeco greedy has work 54 and largest tensor 8. Both reproduce the AD evidence mass. The plan is conditioned by slicing the evidence variables, including retained scalar factors. `runs/{random,greedy}/complexity.json` records independent per-step counts. The plan binds the exact model bytes and evidence, and replay validates both and the tree structure. Random uses an explicitly seeded local LCG, not omeco's RNG. The random plan's largest intermediate has indices T,S,L,B (16 entries, steps 3, 4 and 6). The greedy plan's largest intermediates have indices X,E,B and X,E,L (8 entries, steps 2 and 5).
- Rust synchronous BP stores the beliefs, undamped residual and cumulative arithmetic operation count at each state. `artifacts/bp-{prior,AD,ADX}/run.json` records all nodes and settings. On ADX the residual is 8.225e-11, but the maximum marginal error is 0.010611745. The nonuniform two-node tree test checks the message rule.
- Promedus inputs and official answers are fetched with `./fetch_data.sh` into the ignored `promedus/` directory. Greedy and TreeSA outputs and trees are in `artifacts/promedus-{greedy,treesa}`. Both give log10 Z = -7.80674630458243 and state-1 probabilities 0.1975882582, 0.1497534323 and 0.6175220184 for IDs 59, 102 and 139. They meet the sheet's 1e-10 plan agreement, 1e-4 PR and 1e-5 MAR tolerances. `verify.py` compares to the downloaded answers.
- [Promedus cost chart](evidence/promedus-costs.png) and [BP trace](evidence/promedus-bp.png): at 1,000 sweeps BP is **unconverged** (undamped residual 2.1163e-10 > 1e-10), with max selected-query error 0.38235 versus rounded MAR. It performed 22,473,000 counted message-update multiplications/additions; `/usr/bin/time` recorded 0.37 s including program startup, trace serialization and file I/O on this host. Exact run timings are in each query's result JSON; planning and evaluation are separate. BP computes all beliefs while exact runs query only three, so timing is not an equal-work comparison.
- For scalar Z, independent tree counts are greedy 58,221,225 terms / 524,288 largest generated entries and TreeSA 11,590,416 / 262,144. TreeSA uses omeco `fast` (100 temperatures, 20 iterations, one trial), time-only scoring; the library seeds trial 0 with 42. TreeSA is heuristic, not certified optimal. Query-specific costs are saved in `independent-costs.json`; they can differ greatly from the scalar plan.

## Reproduce

From this directory, run `./fetch_data.sh` to obtain the pinned ASIA and Promedus inputs, then use installed Rust 1.98.1. Python plots require NumPy and Pillow; on this host use `/Users/joshua/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3` in place of `python`.

```sh
cargo test --locked
cargo build --release --locked --bins
for c in prior AD ADX; do
  target/release/infer --model asia.uai --evidence "$c" --out "artifacts/$c"
  target/release/infer --model asia.uai --evidence "$c" --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out "artifacts/bp-$c"
done
target/release/infer --model asia.uai --evidence AD --order random --seed 42 --out runs/random
target/release/infer --model asia.uai --evidence AD --order greedy --out runs/greedy
target/release/infer --model asia.uai --evidence AD --plan runs/greedy/order.json --out runs/replay
target/release/infer --model asia.uai --evidence AD --analyze runs/random/order.json --out runs/analysis
python check_numerics.py
python plot_baseline.py
```

Run `./fetch_data.sh`. For the large benchmark, `ROOT=promedus` and `E=41=1,47=1,26=1,29=1,28=1,14=1,123=1,4=1,147=1`:

```sh
target/release/benchmark "$ROOT/Promedus_14.uai" "$E" greedy artifacts/promedus-greedy
target/release/benchmark "$ROOT/Promedus_14.uai" "$E" treesa artifacts/promedus-treesa
target/release/infer --model "$ROOT/Promedus_14.uai" --evidence-file "$ROOT/Promedus_14.uai.evid" --reference-mar "$ROOT/Promedus_14.uai.MAR" --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/promedus-bp
python audit_plans.py
python verify.py
python build_evidence.py
```

Sources: ASIA `TensorBFS/TensorInference.jl` commit `935ed86f8e473f523cac9461c45a1f470e74f0f9`; Promedus `mroavi/uai-2014-inference-competition` commit `22005844423f5bca62798548df808cc418fc0d7d`. Raw model data are not committed; `fetch_data.sh` fetches the exact pinned files. The original course PDF and local download copies are in `/Users/joshua/Downloads/amat5315/week9`; the benchmark data can be restored with `fetch_data.sh`. The full BP trace is included as `artifacts/promedus-bp/run.json.gz`.

## Interface and limitations

ASIA ordering uses the PDF's `infer --order/--seed/--plan/--analyze` interface. Scalar plans are bound to the model bytes and evidence; invalid or mismatched plans are rejected. Analysis writes costs without contraction. `benchmark` is retained for the larger-model query plans. Figures `posteriors.png` and `joint.png` are generated from Rust enumeration and tenferro outputs; the white markers overlay tensor results. The independent NumPy matrix/trace and enumerated Ising checks are printed by `check_numerics.py` and saved under `evidence/`.

`--reference-mar` is required by generalized Rust BP and reports maximum error against the supplied rounded reference. The benchmark evaluator supports positive factors and rejects a fully zero intermediate; it is intended for these datasets. No commit or push was made at the time of local verification.
