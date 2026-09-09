# Independent development review: Parts 1-4

Scope: the uncommitted Part 4 implementation over its red-test commit aeabce1, plus existing pair/dimer code, tests, rendering helper and Makefile. Reviewer read the approved design, PDF pages 9-14, source and development evidence in a separate agent context. This is not the final fresh-clone review required after Part 5.

## Finding

- P2: the CLI built a lattice before enforcing the maximum atom count. A valid even-square value of 10^18 caused a capacity-overflow panic (exit 101), rather than a clean input error. Smaller oversized values could allocate before rejection.
- Resolution: moved the guard into periodic::lattice before allocation, shared by CLI and library callers. A subprocess regression first reproduced exit 101, then passed with exit 1 and a clear error message. Evidence: part4-review-count-red.txt and part4-review-green.txt. The regression test was committed before the fix at 3e91281; the following Part 4 implementation commit contains the fix.

No other actionable findings were reported in the inspected scope.

## Independently obtained evidence

- NumPy recomputed all 200 default production frames. Maximum stored-energy discrepancy: 5.12e-13; drift: 9.827846953852782e-5; T_speed: 0.47705724493829704; chi2/22: 0.7018545454545454.
- Two default runs produced byte-identical metadata and trajectory files.
- The checker rejected missing/extra frames, incorrect steps/times, wrong array lengths, unwrapped coordinates, altered stored energies and altered raw positions.
- Sequential replacement preserved previous output bytes in history; a failed simulation preserved active output bytes.
- LJ/PBC/cutoff, removal of COM velocity, thermostat schedule and Verlet updates agreed with the design.
- Parent development verification after the fix: all 14 release tests passed. Media QA was performed separately.

Student-run grey boxes, measured performance, Part 5, final fresh-clone review and the narrated recording are still pending.
