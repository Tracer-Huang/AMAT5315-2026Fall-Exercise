# Week 6: Energy-based models — verified exercises

AMAT5315, HUANG QIUSHAN. Course source: [Week 6 learning sheet](https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week6-learning-sheet.pdf), retrieved 2026-10-14 (16 pages). Original materials and course-linked readings live in `/Users/joshua/Downloads/amat5315/week6/`.

All four required parts were executed at the specified full budgets on CPU and strictly audited against the PDF. Git publication was authorized by the user on 2026-10-14 after these checks. This repository delivery comprises source, environment and evidence; no course-portal submission is required. The course p16 asks to show the evidence at the start of Week 7 (2026-10-21); it explicitly says "Nothing is submitted." AIS on p11 is an optional extension and was not run.

## Setup and commands

```sh
cd week6  # from the exercise repository root
uv sync --locked
JAX_PLATFORMS=cpu .venv/bin/python part1.py
JAX_PLATFORMS=cpu .venv/bin/python part2.py
JAX_PLATFORMS=cpu .venv/bin/python part3.py
JAX_PLATFORMS=cpu .venv/bin/python part4.py
.venv/bin/python -m pytest tests -q
.venv/bin/python verify.py
.venv/bin/python strict_audit.py  # requires ignored local raw runs
.venv/bin/python spin_detail.py
.venv/bin/python build_review.py
```

If `uv` is on PATH, use `uv sync --locked`. The existing Week 5 uv executable was reused; dependencies were installed in this week's separate .venv. Python 3.14.7, JAX 0.11.2, NumPy 2.5.3; CPU platform, float32 RBM. `uv.lock` captures all resolved packages; `requirements.lock.txt` is an exported alternative. Part 2–4 refuse to overwrite existing raw run directories. To reproduce a further raw run use `--raw /absolute/path/to/a/new-empty-directory` (a separate output set); evidence figures are rebuilt from that run. Downloads are reused, not repeated unnecessarily. Tests do not rerun full training.

## Data and provenance

`inputs/`, `.venv/`, and `artifacts/` are ignored. Raw downloads, all trained epoch weights, samples, completions, random pattern sets, retrieved states, settings, and logs remain local. `evidence/provenance.json` records exact download URLs, byte sizes and SHA-256; `evidence/dataset-stats.json` records original split shapes, pixel range and every class count. CIFAR is read from its binary tar without filesystem extraction. MNIST image/label ordering remains paired. Galleries use nearest interpolation.

MNIST: 60,000 training and 10,000 test grayscale 28x28 images; digit 1 most frequent, 5 least in both splits. CIFAR-10: 50,000 training and 10,000 test RGB 32x32 images; 5,000/1,000 per class, fractions exactly .1. A balanced gallery shows appearance, not the class prevalence; fractions count every split label.

## Dataset download URLs

- `mnist/train-images-idx3-ubyte.gz`: https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
- `mnist/train-labels-idx1-ubyte.gz`: https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
- `mnist/t10k-images-idx3-ubyte.gz`: https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
- `mnist/t10k-labels-idx1-ubyte.gz`: https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
- `cifar10/cifar-10-binary.tar.gz`: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/cifar-10-binary.tar.gz

## RBM: reconstruction is not generation

`rbm.toml` contains the exact prescribed budget. Its PDF-wrapped `wrong_upper` string was joined into one legal TOML line. RBM: 784 visible bits (threshold >=128), 128 hidden bits. E=-a.v-b.h-v.W.h; F=-a.v-sum softplus(b+v.W). `jax.value_and_grad` differentiates mean(F(v0)-F(stop_gradient(vk))). This is an approximate likelihood gradient, not an NLL measurement. Two models share initialization seed 6601; seed 6602 splits into data-order and Gibbs streams, folded by epoch and reset per model; evaluation seed 6606 resets at initialization and each epoch. Only CD chain length differs. Pure SGD .1, 20 complete epochs, 600 batches/epoch, float32. Visible biases use clipped binarized training frequencies only, never test data.

| Model | Initial Hamming | Epoch 20 Hamming | Training seconds | True lower Dice at 1000 | Wrong lower Dice | Advantage |
|---|---:|---:|---:|---:|---:|---:|
| CD-1 | 0.17176 | 0.04640 | 15.38 | 0.38844 | 0.21156 | 0.17687 |
| CD-10 | 0.17176 | 0.04757 | 92.70 | 0.50871 | 0.25661 | 0.25210 |


The surrogate rises while one-step reconstruction improves. It omits log Z and cannot establish an NLL decrease. Reconstruction starts from held-out images, whereas fresh generation starts from matched fair random bits. Both retain 0/1/10/100/1000-step states. Long-chain blobs, rings and strokes are actual measured output, not clean-digit claims; low reconstruction error alone cannot validate generation. Training CD-k and inference chain length are different quantities.

Completion clamps upper halves exactly at every step; it samples the missing half conditioned on that observation. 64 held-out digits x 8 chains per digit x 2 controls x 2 models; wrong halves come from the next different-label image, wrapping in these 64. Dice is evaluated only on missing ink. Standard errors average chains within each digit first. Both true-minus-wrong advantages exceed .1. Completion runtime: 8.08s. Raw final run: `artifacts/rbm-shared-seed/`. An earlier run with a separately offset Gibbs seed is retained in ignored `artifacts/rbm/` for traceability; final evidence uses a single prescribed training seed root.

### Observed training-k and inference-length comparison

The displayed chains are the first six, without selecting favorable outputs. At one reconstruction step both models largely retain the supplied digits; by 1000 steps CD-1's displayed reconstruction chains 1, 2 and 4 (zero-based) are blank, while CD-10 retains more ink in these particular chains. These are observations of the displayed trajectories, not a universal claim about all samples. Fresh starts are matched white/black noise. At one step both models suppress much of the random background; over 10–1000 steps connected ink regions, loops and malformed strokes replace the isolated initial noise. At 1000 steps both models still produce malformed shapes; longer sampling does not guarantee clean digits. The 64-digit completion comparison supports the stronger conclusion about training k: CD-10's true-half Dice .509 exceeds CD-1's .388 despite nearly equal one-step reconstruction errors. Figures therefore separate the effect of training approximation (CD-k) from evolution under each frozen model.

## Classical Hopfield: crosstalk and finite-size capacity

Hebbian J has zero diagonal. Integer C=N*J preserves exact local-field ties and update signs, so a zero field retains its spin. Each recall uses a fresh random order per sweep and per query, stopping only after a sweep with no flips. Every recall was checked for convergence, final local stability, and nonincreasing energy; the 1000-sweep guard was never reached.

N=100/200/400/800, alpha=.05:.01:.25, 20 independent pattern sets at every point; start from first min(10,P) stored patterns. Seeds: 6630+N*10000+load_index*100+repeat; recall adds 1000000. Raw pattern/final-state/sweep records: `artifacts/hopfield/`, 1,680 sets. Standard errors are across independent sets, not pooled correlated bits. Runtime 133.27s.

Unstable-bit fractions approach Phi(-1/sqrt(alpha)); finite-N dotted curves use Phi(-sqrt((N-1)/(P-1))). The recall state can collapse after a small number of unstable bits trigger further flips. Mean-overlap .9 crossings (linear interpolation between load samples): N=100: 0.18539, N=200: 0.17497, N=400: 0.15554, N=800: 0.14652. They decrease with N and each lies within .02 of the learning-sheet reference. The .138 capacity is an infinite-N limit, not an exact finite-size threshold; no crossing confidence interval is claimed from the coarse grid.

## Modern Hopfield: one attention step

Stored patterns are rows in code (the transpose of X in the sheet). Compute softmax(beta*q@patterns.T)@patterns. The independent library path uses actual `jax.nn.dot_product_attention`, with query q, keys=values=patterns, `scale=beta`, implementation xla. Maximum difference over all 100 queries x 6 P x 7 beta combinations: 1.7285347e-06, below 1e-5.

P=3/10/30/100/300/1000; beta=.001/.003/.01/.03/.1/.3/1; first P MNIST training images in original order, 100 targets cycling through them, lower halves replaced by random spins. Seeds 6640+P_index, recall adds 1000. Success requires sign-output overlap >=.95 and target the nearest stored pattern (first index wins exact nearest-neighbor ties). Raw queries/targets/all outputs: `artifacts/modern/`; runtime 7.86s.

At beta=.3 modern success is P=3: 100%, P=10: 99%, P=30: 100%, P=100: 100%, P=300: 95%, P=1000: 95%; classical success is zero at every P. Small beta averages stored digits, large beta concentrates on one. The common background correlates MNIST patterns and defeats classical recall even at P=3: the random-pattern capacity assumption does not apply. Modern retrieval recalls training images it stored; RBM completion handles held-out images it never saw. These are different tasks. This special query/key/value attention identity does not imply a full transformer is just a stored-digit memory.

## Evidence and producing commands

### mnist-distribution

`JAX_PLATFORMS=cpu .venv/bin/python part1.py`

MNIST: native grayscale examples and full-split class fractions.

![MNIST: native grayscale examples and full-split class fractions.](evidence/mnist-distribution.png)

### cifar10-distribution

`JAX_PLATFORMS=cpu .venv/bin/python part1.py`

CIFAR-10: native RGB examples and balanced split fractions.

![CIFAR-10: native RGB examples and balanced split fractions.](evidence/cifar10-distribution.png)

### binarized-spins

`JAX_PLATFORMS=cpu .venv/bin/python part2.py`

Half-intensity binary digits; ink +1, background -1.

![Half-intensity binary digits; ink +1, background -1.](evidence/binarized-spins.png)

### spin-detail

`JAX_PLATFORMS=cpu .venv/bin/python spin_detail.py`

Enlarged first training digit: native 28x28 spins, background -1 and ink +1.

![Enlarged first training digit: native 28x28 spins, background -1 and ink +1.](evidence/spin-detail.png)

### rbm-architecture

`JAX_PLATFORMS=cpu .venv/bin/python part2.py`

784 visible / 128 hidden bits; only cross-layer couplings.

![784 visible / 128 hidden bits; only cross-layer couplings.](evidence/rbm-architecture.png)

### rbm-training

`JAX_PLATFORMS=cpu .venv/bin/python part2.py`

Matched CD-1/CD-10: pre-update surrogate versus held-out sampled reconstruction.

![Matched CD-1/CD-10: pre-update surrogate versus held-out sampled reconstruction.](evidence/rbm-training.png)

### rbm-cd1-sampling

`JAX_PLATFORMS=cpu .venv/bin/python part2.py`

CD-1 reconstruction and fresh generation, 0/1/10/100/1000 steps.

![CD-1 reconstruction and fresh generation, 0/1/10/100/1000 steps.](evidence/rbm-cd1-sampling.png)

### rbm-cd10-sampling

`JAX_PLATFORMS=cpu .venv/bin/python part2.py`

CD-10 reconstruction and fresh generation, matched starts.

![CD-10 reconstruction and fresh generation, matched starts.](evidence/rbm-cd10-sampling.png)

### rbm-cd1-completion-true

`JAX_PLATFORMS=cpu .venv/bin/python part2.py`

CD-1 completion: true upper halves, two chains for each displayed digit.

![CD-1 completion: true upper halves, two chains for each displayed digit.](evidence/rbm-cd1-completion-true.png)

### rbm-cd1-completion-wrong

`JAX_PLATFORMS=cpu .venv/bin/python part2.py`

CD-1 control: wrong-label upper halves, same target lower half.

![CD-1 control: wrong-label upper halves, same target lower half.](evidence/rbm-cd1-completion-wrong.png)

### rbm-cd10-completion-true

`JAX_PLATFORMS=cpu .venv/bin/python part2.py`

CD-10 completion: true upper halves.

![CD-10 completion: true upper halves.](evidence/rbm-cd10-completion-true.png)

### rbm-cd10-completion-wrong

`JAX_PLATFORMS=cpu .venv/bin/python part2.py`

CD-10 control: wrong-label upper halves.

![CD-10 control: wrong-label upper halves.](evidence/rbm-cd10-completion-wrong.png)

### rbm-completion-overlap

`JAX_PLATFORMS=cpu .venv/bin/python part2.py`

Missing-half ink Dice; standard errors over 64 digit-level chain means.

![Missing-half ink Dice; standard errors over 64 digit-level chain means.](evidence/rbm-completion-overlap.png)

### hopfield-unstable

`JAX_PLATFORMS=cpu .venv/bin/python part3.py`

Unstable bits versus Gaussian crosstalk; dotted finite-N corrections.

![Unstable bits versus Gaussian crosstalk; dotted finite-N corrections.](evidence/hopfield-unstable.png)

### hopfield-capacity

`JAX_PLATFORMS=cpu .venv/bin/python part3.py`

Mean final overlap and fraction >=.9; 20 independent sets per point.

![Mean final overlap and fraction >=.9; 20 independent sets per point.](evidence/hopfield-capacity.png)

### attention-weights

`JAX_PLATFORMS=cpu .venv/bin/python part4.py`

P=100 softmax weights at beta .003/.03/.3.

![P=100 softmax weights at beta .003/.03/.3.](evidence/attention-weights.png)

### modern-beta-gallery

`JAX_PLATFORMS=cpu .venv/bin/python part4.py`

One-step retrieval across every beta versus classical recall.

![One-step retrieval across every beta versus classical recall.](evidence/modern-beta-gallery.png)

### modern-success

`JAX_PLATFORMS=cpu .venv/bin/python part4.py`

100 targets per P; all beta success curves with classical control.

![100 targets per P; all beta success curves with classical control.](evidence/modern-success.png)

### modern-retrieval-gallery

`JAX_PLATFORMS=cpu .venv/bin/python part4.py`

P=100 target/query/classical/modern beta .3 comparison.

![P=100 target/query/classical/modern beta .3 comparison.](evidence/modern-retrieval-gallery.png)

## Final verification: distributions and sampling

The real target distribution p_data is unknown; neither the image galleries nor finite class fractions specify it exactly. Part 1's empirical distribution assigns weight 1/N to each original record; the full-split bar plots measure its label frequencies. The fitted RBM p_theta(v) is proportional to exp(-F_theta(v)); its parameters define a probability model distinct from the finite dataset. Binary block Gibbs is the sampling algorithm, not the model itself. Part 2's trajectories show that a finite chain need not have mixed, and a short reconstruction can retain its start. CD approximates the model expectation in the likelihood gradient by a data-initialized k-step chain endpoint, holding that endpoint fixed during differentiation. Clamping changes this chain to conditional inference on missing pixels. Part 3 checks the Gaussian crosstalk and finite-size loss of recall; Part 4's beta-dependent weights and image outputs connect modern retrieval directly to the actual attention vector.

## Verification and review

`AUDIT.md` maps every PDF DO/VERIFY/Wrap-up requirement to source and evidence. `evidence/strict-audit.json` records full raw-array recomputation and trajectory replay. Raw settings snapshots live beside each measured run and are generated on future runs. `evidence/verification.json` maps finite acceptance checks to the measured outputs; `evidence/unit-tests.txt` records nine passing model and visualization tests. Free energy is checked by exact enumeration, the CD gradient by a NumPy correlation formula; other tests verify stop-gradient, sampled conditional frequencies, binary clamping, energy/stability/ties and JAX attention equivalence. Native-image and plot layout were visually inspected. See `STUDY_NOTES.zh.md` for the full Chinese lesson explanation and `review.html` for the local review gallery. Downloaded readings are resources for your own study, not a claim that your personal reading or oral explanation has been completed.
