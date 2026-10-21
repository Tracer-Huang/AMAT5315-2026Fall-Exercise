# Week 7: Diffusion and flow matching

HUANG QIUSHAN · AMAT5315 Fall 2026. Parts 1–5 implemented, run and checked against the 16-page Week 7 PDF. The initial run's three omissions have been repaired: early loss history is visible, both stochastic repeats are plotted, and the original full-square velocity RMS criterion is now met at all three times.

Start with [review.html](review.html), [Chinese study notes](STUDY_NOTES.zh.md), and the [strict PDF audit](STRICT_AUDIT.zh.md). The course asks to **push source, environment and evidence**, then show them at the start of Week 8 (scheduled 2026-10-28); the PDF states **“Nothing is submitted”**, with no separate LMS submission. The user's later instruction authorizes this push once the work passes strict checking. Earlier commits were deferred under the original review-first instruction.

## Reproduce

From this directory, with Python 3.14 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
uv run python -m pytest tests -q
uv run python run_toy.py
uv run python diagnostics.py
uv run python download_model.py
uv run python run_images.py
uv run python verify.py
uv run python strict_audit.py
uv run python build_review.py
```

The completed local environment also supports `uv run --offline`; this session uses `.bootstrap/bin/uv` because uv is not on PATH. `uv.lock` pins all dependencies. The independent Week 7 environment was bootstrapped from an APFS clone of Week 6 packages and reconciled with uv; Week 6 was read only. `evidence/environment.json` records versions. All model computations use CPU JAX.

`run_toy.py` trains missing checkpoints and reuses existing ones. The selected configuration is 100000 updates for **each** of two independently initialized networks, batch 512, width 64, Adam, initial LR 0.001, cosine decay to exactly zero, seed 7101. Ten saved stages include initialization and the final model. Training targets are fresh random added noise or fresh random straight-path velocity, never an exact-predictor teacher. Settings and every update's loss are inside each raw training NPZ. To start a different run, first archive the current `artifacts/` and `evidence/`; do not overwrite an unversioned result. The first failed run is retained locally under `artifacts/baseline-v1/`.

The original materials are in `/Users/joshua/Downloads/amat5315/week7/`: PDF, extracted text, original resource ZIP and extracted files. On another machine, extract `week7/data/images.npz` from the [official ZIP](https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week7-resources.zip) to `inputs/images.npz`, or use the PDF's direct `week7-images.npz` link; the two NPZ files were byte-identical here. For the real-image gallery, `run_images.py` reuses Week 6's `data.cifar('test')` and needs its documented local CIFAR archive at `../week6/inputs/cifar10/cifar-10-binary.tar.gz`. Other Parts 4–5 operations require only the Week 7 images and pinned checkpoint.

## Learned network and the repaired velocity field

The four equally weighted target means are (3,0), (0,3), (-3,0), (0,-3), each covariance 0.09I. Under the fixed orthogonal coordinates

```
u = (x+y)/sqrt(2),  v = (x-y)/sqrt(2),
```

the four means become all sign pairs `(±3/sqrt(2), ±3/sqrt(2))`. The two scalar marginals are independent and identical. We therefore use the **same shared odd scalar predictor** on both coordinates, then transform back. This uses the target's symmetry; it does not use a conditional-mean oracle.

The predictor is `f(q,s)=a_theta(s) q + b_theta(s) tanh(c_theta(s) q)`. A three-hidden-layer SiLU time MLP of width 64 maps 16 sinusoidal time features to three learned coefficients, with `c_theta(s)=softplus(raw_c_theta(s))/(s+0.05)`. All coefficients start randomly and are fitted by the requested stochastic regression loss. The linear term gives a sensible far-field function class; the odd tanh term represents the two modes. Noise and velocity use the **identical architecture**, initialized and optimized separately. This is a deliberate architecture change from the initial generic 2D MLP, not a substitution of the exact velocity/noise. `core.exact_noise` and `core.exact_flow` are used only for evaluation and the explicitly labelled exact-sampler comparisons.

The original full `21×21` grid over `[-4,4]^2`, equal weighting of every point, is unchanged. Its RMS errors changed from **0.3442,0.3009,0.0725** to **0.00180,0.00176,0.04773** at s=0.1,0.5,0.9. No density mask or clipped arrow lengths enter those RMS values. A denser full 41×41 diagnostic also passes: 0.02249,0.00184,0.04550. Density-restricted diagnostics remain explicitly labelled supplementary results.

## Sampling and statistical protocol

- Diffusion: 1000 mathematical levels; beta linearly 0.0001…0.02; alpha-bar includes level zero = 1. DDPM variance is beta, no noise on the final step, no mixture clipping. DDIM uses the clean-estimate update, not DDPM with randomness removed.
- P1: 2000 exact/learned samples; true forward Markov paths; ten training checkpoints share the same ten Gaussian starts and reverse noises. Every trained frame displays its available loss history, including update 1. Raw paths outside the fixed figure bounds are retained.
- P2: 2000 shared starts; DDIM runs with seeds 11/12 agree; DDPM runs use distinct reverse-noise draws. Both runs are shown in distributions and paired trajectories, and saved in `artifacts/paired-runs.npz`.
- P3: fresh flow regression on `(1-s)x0+sz`, uniform s, target `z-x0`; Euler integrates from s=1 down to zero. Compare DDPM/DDIM/flow at 1,2,5,10,20,50,100 steps, plus 1000 for DDPM/DDIM: 23 runs, N=4000 each. Exact predictors have a separate labelled sweep.
- Cluster assignments use nearest prescribed centre. Save fractions, means and coordinate standard deviations; centre error is the maximum Euclidean displacement. Energy distance is the unbiased U statistic excluding same-set diagonal pairs. Its negative finite-sample values remain in JSON, with a display floor of 1e-4 only on log figures. Null scale: 20 independent exact-set pairs, N=4000 per set, 95% quantile 0.004683. Fraction band: two binomial standard errors. These are run-specific fluctuations, not universal tolerances.

## Evidence and producing commands

| Evidence | Command | Purpose |
|---|---|---|
| [forward.png](evidence/forward.png), [forward.gif](evidence/forward.gif) | `uv run python run_toy.py` | Same 2000 points and fixed axes; print/save endpoint moments. |
| [loss.png](evidence/loss.png), [training.gif](evidence/training.gif) | `uv run python run_toy.py` | All ten stages, ten shared paths, complete early loss histories. |
| [ddpm.png](evidence/ddpm.png), [reverse.gif](evidence/reverse.gif) | `uv run python run_toy.py` | 2000-point distribution and reverse movie. |
| [determinism.png](evidence/determinism.png), [trajectories.png](evidence/trajectories.png) | `uv run python run_toy.py` | Both DDIM and both DDPM runs; paired paths use solid/dashed lines. |
| [velocity.png](evidence/velocity.png) | `uv run python run_toy.py` | Original full-grid learned/exact fields and unweighted RMS. |
| [comparison.png](evidence/comparison.png) | `uv run python run_toy.py` | Energy, maximum fraction error and maximum width error vs steps. |
| [velocity-supported.png](evidence/velocity-supported.png), [exact-vs-learned.png](evidence/exact-vs-learned.png) | `uv run python diagnostics.py` | Supplemental declared-domain and exact-predictor diagnostics. |
| [photographs.png](evidence/photographs.png), [real-vs-generated.png](evidence/real-vs-generated.png) | `uv run python run_images.py` | Four generated images, reverse stages, real CIFAR gallery. |
| [inpainting.png](evidence/inpainting.png) | `uv run python run_images.py` | Four images, four completions each, all MSEs and variance maps. |

## Results and interpretation

Forward endpoint mean (0.0269,-0.0149), variance (1.0395,1.0907): within 0.1 of zero/one. P1 DDPM fractions are 0.2425,0.2675,0.2375,0.2525; maximum centre displacement 0.03673, maximum width error 0.01849. All meet the PDF's P1 tolerances.

DDIM repeat difference is exactly **0**; two DDPM runs from shared starts land in the same cluster **23.85%** of the time. DDIM fraction error is 0.014, below 0.015. Finite-sample widths are 0.282–0.307, within ±0.03 of 0.3 but not a literal reproduction of the reference run's descriptive 0.29–0.32 interval. Reference numbers are not our measured numbers.

The 20-step flow sampler has energy distance **0.002251**, maximum width error **0.01299**: meets <0.005 and <0.03. Its centre displacement is about 0.066 at this coarse Euler step size and decreases with finer grids; do not apply the Part 1 DDPM centre threshold to a different coarse-step experiment or claim exact transport. All 23 runs retain complete moment measurements. One/two-step samplers fail, and exact-predictor comparisons expose finite-step error. This toy example does not prove a universal runtime ranking. Network calls compare sampling effort, excluding training cost.

Photo structure emerges late. Fixed pretrained weights learned the photographic prior before download; this code only evaluates them and executes 1000 DDPM steps. Four 32×32 outputs have animal/bird/scene-like structure; their exact semantic categories remain visually ambiguous at this resolution.

RePaint produces 16 completions with mean hidden MSE **0.20936**, observed-pixel maximum change **0**, outside-hole variance **0**. Inside-hole variance is positive. Different completions can be plausible while disagreeing with the one withheld truth; variance measures disagreement, MSE measures error against that truth.

## Frozen photographs model and RePaint details

`google/ddpm-cifar10-32`, revision `267b167dc01f0e4e61923ea244e8b988f84deb80`; `config.json` and `diffusion_pytorch_model.safetensors`. The pinned official LFS SHA256 is `1558dbe3fb093b0857d473e376b711eeda1b186c557be5e85b3a56300cd49f0f`, verified. `download_model.py` downloads resumable byte ranges and checks this identity. It obtains the same files as the PDF's pinned Hugging Face command.

JAX U-Net is source-reviewed against [diffusers v0.40.0](https://github.com/huggingface/diffusers/tree/v0.40.0): public NCHW float32 interface, internal NHWC, 32-group norms with config epsilon, single-head attention for null attention_head_dim, right/bottom downsample padding, nearest-neighbour upsampling, pinned time embedding. Legacy query/key/value/proj_attn names are loaded directly. Mathematical t maps to checkpoint t-1. No photo-state clipping, only display clipping. Starts use seeds 7400…7403; DDPM reverse seeds 7410…7413. Cross-framework PyTorch numerical parity was **not** performed and is not claimed; structure/source review, parameter identity, generated outputs and sampler behavior were checked.

RePaint: test split only; hide rows/columns 6:26 across all channels; pass masked observations and mask only. 50 reverse jumps, two attempts except last, forward noise between attempts, **99 actual model calls per seed batch**. Use the posterior jump variance and destination-noised observations; clip only clean estimate. Seeds 7201…7204 each operate on a batch of four images. Ground truth is used only afterwards for the 1200-value hidden-pixel MSE. Runtime callback instrumentation verifies the actual 99-call level sequence rather than trusting a returned constant.

## Validation and delivery boundary

[Strict audit JSON](evidence/strict-audit.json) recalculates raw moments/energy, replays paired samplers, inspects actual frame/trajectory artists, checks artifact producer hashes, evaluates the unchanged full grid, and checks inpainting and model identity. `verify.py` and `strict_audit.py` both fail with nonzero exit status if their acceptance checks fail. A regression test injects a failed metric into a temporary copy and proves this failure blocks success.

All current PDF acceptance checks pass. `artifacts/baseline-v1/` retains the earlier failed results; raw runs, model downloads and environments remain out of Git. Final course delivery uses the existing repository after verification, under the user's conditional authorization; it does not include an LMS upload.
