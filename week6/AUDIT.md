# Week 6 strict PDF audit — 2026-10-14

Result: **all required code, numerical experiments, evidence and written explanations pass**. The user authorized commit/push after strict verification. PDF p16 requests pushing source, environment and evidence; it explicitly requires no separate assignment submission. Student reading, oral explanation and showing evidence at the beginning of Week 7 remain personal activities and are not claimed as completed. AIS (p11) and N=1600 (p13) are explicitly optional and were not run.

## Authoritative source

The complete 16-page Week 6 PDF was read, including all DO, VERIFY, Final verification and Wrap-up blocks. The live official PDF was re-downloaded during this audit and has the same SHA-256 as the saved source: `1ada979d931cde1877e7d826034de3eec9b902acd0e30e953643983a7ff4cfb5`.

Source: https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week6-learning-sheet.pdf

## Requirement coverage

| PDF page / requirement | Source / actual evidence | Finding |
|---|---|---|
| p2: week6/, uv, CPU; figures in evidence, raw runs/settings in artifacts; exclude raw downloads/runs | pyproject.toml, uv.lock, requirements.lock.txt, .gitignore, evidence/environment.json | Pass. Raw settings snapshots are retained beside the measured arrays and generated automatically on future runs. |
| p2: original MNIST/CIFAR downloads; retain exact URLs in README | data.py, README Dataset download URLs, evidence/provenance.json | Pass. All five files' sizes/hashes checked against provenance; original splits retained. |
| p2: reusable grayscale/RGB gallery, optional labels, split frequency plotting, native pixels | plotting.py gallery, distribution_figure, class_frequencies; tests/test_plotting.py | Pass. Both grayscale/RGB tested; all labels counted independently of gallery selection; nearest interpolation. |
| pp3–4: one labeled-gallery/frequency figure per dataset; split counts and representations | evidence/mnist-distribution.png, cifar10-distribution.png, dataset-stats.json | Pass. MNIST 60000/10000, 28x28 uint8; CIFAR 50000/10000, 32x32 RGB uint8. MNIST 1 most/5 least in both splits; CIFAR every class .1. |
| pp4–5: exact rbm.toml, 128 hidden, >=128 binarization, energy/free energy/binary block Gibbs | rbm.toml, models.py, part2.py; tests/test_models.py | Pass. Every training/initialization/evaluation/completion TOML field checked against the PDF; line-wrapped wrong_upper string joined without altering meaning. |
| p5: train CD1/CD10 with stop-gradient; matched initialization, order, evaluation randomness; record diagnostics | part2.py, evidence/rbm-metrics.json, artifacts/rbm-shared-seed/ | Pass. 20 full epochs x 600 batches x 100 images, plain SGD .1, float32. Root 6602 split into data/Gibbs streams and reset per model. |
| p8 VERIFY1: spins and RBM topology | evidence/binarized-spins.png, spin-detail.png, rbm-architecture.png | Pass. First 16 original training images, one enlarged 28x28 spin grid; ink +1/background -1; only visible-hidden edges. |
| pp8–9 VERIFY2: surrogate/history, sampled held-out Hamming, below half initialization | evidence/rbm-training.png; independent replay of all 42 checkpoint evaluations | Pass. CD1 0.171755 → 0.046397; CD10 0.171755 → 0.047573. Both below 0.085878. Surrogate correctly identified as not NLL. |
| p9 VERIFY3: frozen reconstruction versus fresh generation; matched starts; 0/1/10/100/1000 frames | evidence/rbm-cd1-sampling.png, rbm-cd10-sampling.png; raw 5x64x784 arrays | Pass. Every stored trajectory replayed exactly with fixed seeds. First six chains shown, no favorable selection; malformed strokes/blobs honestly described. |
| p10 VERIFY4: 64 held-out digits, 8 chains, correct and next-different-label wrong clamps, requested steps | four completion galleries, raw 5x1024x784 arrays, rbm-completion-overlap.png | Pass. All saved observed halves identical to their selected observation; every full clamped trajectory replayed exactly. |
| p10: missing-half ink Dice, errors over 64 digits, true-minus-wrong >=.1 | independent NumPy Dice and digit-level SE recomputation | Pass. CD1 true/wrong .388438/.211564, advantage .176874; CD10 .508714/.256610, advantage .252104. |
| p11: random patterns, zero diagonal Hebbian rule; asynchronous random order; h=0 ties retained; convergence | models.py recall; part3.py; model tests and all raw finals | Pass. C=N*J integer fields are equivalent for zero-temperature dynamics and avoid roundoff at ties. Every raw final is stable; energy never increases; max sweeps below guard. |
| p11: full N/load/repetition/first-pattern sweep; seeds/runtime | evidence/hopfield-metrics.json, 1680 raw pattern-set files | Pass. N=100/200/400/800; .05:.01:.25; 20 independent sets per point; first min(10,P) starts. All raw summaries independently recomputed. |
| pp12–13 VERIFY1: unstable fraction/error bars/Gaussian comparison | evidence/hopfield-unstable.png | Pass. Errors over independent sets; large-N and finite-N theory both shown. N800 alpha .2 observed .0123547, predicted .0126737. |
| p13 VERIFY2: mean final m, fraction m>=.9, alpha .138 mark, decreasing crossings within .02 reference | evidence/hopfield-capacity.png | Pass. Crossings .185393/.174967/.155539/.146520; reference .186/.170/.156/.150. Finite grid interpolation not claimed as an exact critical point. |
| pp13–14: first P stored MNIST spins; one modern update and actual JAX attention; 100 cyclic targets, lower-half noise; all P/beta | part4.py, models.py; six full raw files | Pass. 6 P x 7 beta x 100 queries; same patterns/queries for classical and modern; explicit scale=beta. |
| p15 VERIFY1: weight plots at three beta values; largest attention difference <1e-5 | evidence/attention-weights.png, modern-metrics.json; strict replay output | Pass. Actual JAX max difference 1.7285347e-6. Separate NumPy softmax maximum discrepancy 6.1038445e-6. |
| p15 VERIFY2: queries across beta beside classical output | evidence/modern-beta-gallery.png | Pass. All seven beta values; small beta averages, large beta concentrates; same sparse classical output contrasted. |
| pp15–16 VERIFY3: success versus P; P100 target/query/both-retrieval gallery; beta .3 >=.90, classical 0 | evidence/modern-success.png, modern-retrieval-gallery.png | Pass. Beta .3: 100/99/100/100/95/95%; classical 0% at every P. Both overlap and nearest-target requirements recomputed from pixels. |
| p16 Final verification: distribution/model/sampler distinction, approximate CD gradient, clamping, crosstalk/capacity, attention/beta | README final verification and preceding sections; STUDY_NOTES.zh.md; figures | Pass. Explanations grounded in the measured results and mathematical assumptions. |
| p16 Wrap-up: every figure beside producing command, provenance, setup/settings, push source/environment/evidence | README covers all 19 evidence figures; environment/config/lock/source included | Prepared and verified for the authorized Git publication. Downloads/raw arrays, .venv and internal TASK.md remain excluded. |

## Checks performed and their limits

- `uv sync --locked`: reproducible environment record checked.
- `python -m pytest tests -q`: nine model/visualization tests pass. Free energy is independently checked by enumerating hidden states; CD parameter gradient by NumPy correlations; stop-gradient, conditional probabilities, binary clamping, ties/energy and library attention are covered.
- `python verify.py`: 19 finite PDF acceptance checks pass; the evidence inventory now contains 19 figures.
- `OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 JAX_PLATFORMS=cpu python strict_audit.py`: original data identity, every saved RBM epoch evaluation, every saved Gibbs/completion frame, all 1680 random pattern sets and all modern outputs checked; direct NumPy statistics agree with measured summaries. See evidence/strict-audit.json for detailed checks and runtime.
- Every final figure decoded; revised native gallery, spin detail and existing numerical figures inspected for meaning/layout. Every local review image/link resolves. No numerical thresholds or invariants were weakened.
- The full expensive training need not be repeated: training code, prescribed config and measured weights are unchanged; exact seeded inference replay and raw-statistic recomputation close the relevant audit gaps. Source changes add reusable plotting, raw setting snapshots and printed attention maximum; they do not change trained models or recall outputs.

## Repairs made before Git publication

1. Added direct dataset URLs to README, not only to provenance JSON.
2. Extracted reusable gallery/frequency functions and tested grayscale/RGB and full-label fractions.
3. Added the enlarged native spin plot matching the reference explanation.
4. Placed existing run settings next to raw arrays; future runners save these snapshots themselves. The audit-created copies are sourced from each recorded run's existing settings, not newly inferred parameters.
5. Added concrete observed training-k/inference-length comparisons and the complete English final-verification explanation.
6. Replaced outdated pending-review text after the user's explicit publication authorization; kept original datasets, raw runs and earlier trials local.

The original instruction to withhold submission overrode p2's incremental commit timing during initial work. Following the subsequent approval, the completed work is committed in task-scoped commits, then the strict audit and final wrap-up, and pushed. Existing unrelated Week4/Week5 untracked work is preserved.
