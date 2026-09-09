# Final independent review

Reviewed a clean GitHub clone at `44a49f41c4ea083f325663924e8fa5ff3a0d24b7`, against the Part 1-5 designs/plans and the supplied learning sheet. A fresh reviewer inspected actual code, tests, raw trajectories, timing trials and compressed profiles. This review was separate from the earlier Part 1-4 development review.

## Findings and resolution

| Priority | Finding | Resolution | Commit |
| --- | --- | --- | --- |
| P2 | The reproduction guide called default `md run` while labeling baseline timing/profile as naive; cells is now the default. | **fixed**: baseline/profile commands explicitly select `--force naive`; the separate cells profile is documented and the checkpoint text is current. | `64245ae` |
| P2 | Fresh-clone setup installed only Python dependencies, while later commands used a potentially absent/stale bare `md`. | **fixed**: setup installs this checkout with `cargo install --path md --locked --force`, puts Cargo's bin directory first on PATH, and checks the command/version. | `64245ae` |
| P2 (earlier development review) | Excessive atom count could allocate before the limit was checked. | **fixed**: the guard is in `periodic::lattice` before allocation; a subprocess regression verifies a clean exit 1 rather than a capacity-overflow panic. | `f4e96b2` (red regression: `3e91281`) |

No additional Rust correctness defect was found within the reviewed scope. The final two fixes affect documentation only; the fresh-clone code/tests and physical results remain unchanged.

## Verified evidence

- All **21 release tests passed** in the fresh clone; `make reproduce` and `check artifacts` completed with **PASS**. Receipts: `evidence/fresh-clone-*.txt`.
- Independent NumPy recomputation of all 200 default frames: drift `9.8278469539e-5`, T_speed `0.4770572449`, chi2/22 `0.7018545455`; maximum stored-energy discrepancy `5.12e-13`.
- All 200 heating frames reproduced stored energies within `5.00e-12`; first/last T_speed were `0.2044875` and `1.197`.
- Actual profile stacks and symbol sidecars reproduced `105/109` naive force samples and `77/84` cells samples, with process lifetimes `0.110848 s` and `0.084785042 s`.
- Every timing median/min/max matched its raw trials. The measured scaling advantage is supported; the N=100 slowdown is disclosed.
- Cold/hot raw data reproduced mean-square displacements `0.031704 / 23.324234` and contrasts `0.575850 / 0.197121`. Heating contrast reproduced `0.332996 -> 0.096009`.
- The supplied NumPy baseline is byte-identical to the repository copy.

The code/evidence review did not substitute for browser or media QA. Those receipts are recorded separately in `evidence/media-qa.json` and the page/recording evidence. The screen demonstration is silent at the student's request; it must not be described as a narrated recording.
