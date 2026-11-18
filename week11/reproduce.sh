#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pred --version
cargo build --release --locked
tool=target/release/tnsample
mkdir -p runs/pred artifacts
pred path Factoring SpinGlass > runs/pred/path.txt
for target in 15 6 7; do
  n=2
  if [[ $target == 15 ]]; then n=3; fi
  prefix=runs/pred/factor_$target
  pred create Factoring --target "$target" --m 2 --n "$n" -o "$prefix.json"
  pred reduce "$prefix.json" --to SpinGlass -o "$prefix.bundle.json"
  pred solve "$prefix.bundle.json" --json > "$prefix.solve.json"
done
pred evaluate runs/pred/factor_15.json --config 1,1,1,0,1 --json > runs/pred/factor_15.evaluate.json

"$tool" --mode gadgets --seed 2026 --out runs/gadgets
"$tool" --mode count --l 10 --p-af 0 --beta 2 --seed 2026 --out runs/count_p0
"$tool" --mode count --input runs/pred/factor_15.bundle.json --beta 2 --seed 2026 --out runs/count_f15
"$tool" --mode sample --input runs/pred/factor_15.bundle.json --beta 2 --samples 2000 --seed 2026 --out runs/sample_f15
configuration=$("${PYTHON:-python3}" -c '
import json
samples=json.load(open("runs/sample_f15/run.json"))["evidence"]["samples"]
print(",".join(map(str,min(samples,key=lambda s:s["energy"])["bits"])))
')
pred extract runs/pred/factor_15.bundle.json --config "$configuration" --json > runs/pred/factor_15.extract.json
for k in 2 3 4 5 6 7 8; do
  target=$(( ((1 << k) - 1) * ((1 << k) - 1) ))
  prefix=runs/pred/scale_$k
  pred create Factoring --target "$target" --m "$k" --n "$k" -o "$prefix.json"
  pred reduce "$prefix.json" --to SpinGlass -o "$prefix.bundle.json"
  for order in greedy treesa; do
    "$tool" --mode plan --order "$order" --input "$prefix.bundle.json" --seed 2026 --out "runs/plan_k${k}_$order"
  done
done
for n in 4 5 6 7 8 9 10 11 12; do
  "$tool" --mode count --n "$n" --seed 2026 --out "runs/count_nq$n"
done
for n in 6 8; do
  "$tool" --mode sample --n "$n" --samples 200 --seed 2026 --out "runs/sample_nq$n"
done
"${PYTHON:-python3}" review.py
