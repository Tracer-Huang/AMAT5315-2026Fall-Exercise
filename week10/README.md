# Week 10: Approximate tensor contraction for the Ising model

This project studies SVD truncation and the zero-field square-lattice Ising model, H = -sum(s_i s_j), J = k_B = 1. The boundary MPS method uses open edges; TRG uses periodic edges and a power-of-two side length. Both report g = log(Z)/L². The numerical results and acceptance status are in `artifacts/` and `runs/`; these generated files are intentionally ignored by Git.

## Environment and source

macOS on Apple Silicon, Rust/Cargo, system Accelerate LAPACK and BLAS, Python 3, NumPy and Matplotlib. `Cargo.lock` records Rust dependencies. The TRG splits and regrouping follow the course algorithm and the specified [reference](https://github.com/GiggleLiu/BackwardsLinalg.jl/blob/b16273a2d695fef31334085e40f9e0341a1ec0f0/examples/trg.jl). For random matrices, rand_chacha ChaCha8Rng with rand_distr StandardNormal and seed 2026 provides the generated entries. SVD uses Accelerate `dgesvd`; `dgesdd` produced an inaccurate successful result in an independent 16×16 transfer-matrix comparison and is not used.

## Build, run and verify

From `week10/`:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cargo test --release
cargo build --release --bins
cargo run --release --bin week10
.venv/bin/python verify_svd.py
mkdir -p runs
# Each output directory must not already exist; use a new name for each run.
target/release/thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/my-boundary64
target/release/thermo --method trg --l 1024 --k 0.44 --chi 16 --out runs/my-trg1024
.venv/bin/python verify_thermo.py
.venv/bin/python -u scan.py
.venv/bin/python audit_evidence.py
```

Alternatively, `cargo install --path . --root .install --locked` installs both binaries locally; run `.install/bin/thermo` with the same arguments. `scan.py` reuses complete evidence files and does not overwrite them; remove nothing automatically. If a run was interrupted and left an incomplete output directory, choose or move that directory aside before resuming. Arrays in the SVD experiment are little-endian f64, column-major; `verify_svd.py` independently reconstructs errors from saved arrays.

Boundary evidence includes the final MPS tensors in `(left, physical, right)` order, logarithms of per-row normalization scales and discarded squared singular weights. Contract the physical index 0 with `b[0]=1/sqrt(cosh K)` at each site, then add the stored log scales to the logarithm of the final scalar. TRG saves the four split factors and both spectra at each step, the log scale and tensor dimensions, and the final tensor in `(r,u,l,d)` order. Multiply each scale's logarithm by the number of tensors in that step and add the logarithm of the final opposite-leg trace. `audit_evidence.py` independently checks those reconstructions. `exact_transfer_check.py` independently computes the open 16×16 transfer matrix at selected temperatures.

`scan.py` evaluates K=0.40–0.50 in steps of 0.0025 at χ=16 for each size; around each nearest coarse peak it evaluates nine points separated by 0.00125 at χ=16 and 32. Specific heat is computed by centered second differences of g, then the maximum and its two neighbours are fitted by a parabola in K. It compares χ=32 estimates using the refined grid and every other point, fits T_peak against 1/L using all three sizes and the largest two, and saves tables and plots. Open and periodic sizes are respectively (16,32,64) and (64,256,1024). `verify_thermo.py` independently enumerates spin configurations and checks the specified g and energy-variance tolerances, including parallel bonds on the periodic 2×2 lattice.

## Results and checks

| Geometry | Sizes | T_peak, χ=32 | T_c, all sizes | T_c, largest two |
|---|---|---|---:|---:|
| Open | 16, 32, 64 | 2.093105, 2.179676, 2.224220 | 2.267506 | 2.268765 |
| Periodic | 64, 256, 1024 | 2.281602, 2.272260, 2.270012 | 2.269205 | 2.269263 |

The Onsager value is 2.269185314. Both three-size fits are within 1%, and peak shifts under the specified rank and step changes are within 1%. The χ=16/32 comparison, step comparison and summed solver timings appear in `artifacts/peak-summary.csv`; its timing sums unique runs and represents total solver time, not elapsed wall time. Detailed values are in `peak-results.json`, `g-timings.csv`, `small-checks.json` and `acceptance.json`. Rebuild the table with `.venv/bin/python summarize_results.py` and run `.venv/bin/python verify_results.py` to repeat the acceptance audit. The SVD matrix reaches 10% relative error at rank 395, requiring 1.543 times the original storage. The optional peak-height extension is not included.
