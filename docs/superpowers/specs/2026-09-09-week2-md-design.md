# Week 2 MD design for review

Source of requirements: the 20-page `week2-learning-sheet.pdf` supplied by the student, read on 2026-09-09. All units are reduced and all particle vectors have two components.

Workspace: `/Users/joshua/Downloads/amat5315/Repositorie/AMAT5315-2026Fall-Exercise`.
Publication destination: `Tracer-Huang/AMAT5315-2026Fall-Exercise`, existing repository.

## Part 3: force to motion

- A `System` owns position, velocity, and cached acceleration vectors. Measurement functions borrow the system; integration exclusively borrows it mutably.
- `Integrator::step` is implemented by `ForwardEuler` and `VelocityVerlet`. One generic driver accepts either implementation. Euler uses old velocities and accelerations for both updates; Verlet uses kick-drift-force-kick and retains the new accelerations.
- An explicit force model separates the open-boundary, plain LJ dimer from the later periodic, shifted-cutoff fluid. Pair distances of zero or non-finite states are errors rather than silently softened interactions.
- Dimer initial positions: (0,0), (1.2,0); zero velocities; dt=0.01. Both methods run 500 steps from the same initial state through the same driver. Tests require Verlet maximum relative energy error <1e-3 and Euler final relative error >0.5. A 5000-step Verlet run must remain bounded.
- Rust examples export measurements from the actual library. A small Python plotting helper creates field.png and dimer.png from those measurements. The long dimer panel displays relative error multiplied by 1000.

## Part 4: equilibrium CLI

- `md run`, `md check`, and `md video` share the tested physics library. Serde provides JSON; a small, documented CLI layer validates parameters. Rendering can call a Python plotting helper and ffmpeg without moving the physics loop out of Rust.
- Default flags: n=100, rho=0.8, temperature=0.5, dt=0.01, eq_steps=2000, steps=10000, sample_every=50, seed=2026, integrator=velocity-verlet.
- Even-sided square counts (100,400,1600) form periodic triangular lattices. Box sides must both exceed 2*rc, rc=2.5. Minimum-image displacement, position wrapping, and a potential shift preserve the specified physical model. The force is not force-shifted.
- Gaussian initialization uses a documented seeded generator, removes centre-of-mass velocity, and rescales to T_thermo=2K/(2N-2). Rescaling occurs initially and every 50 equilibration steps; ordinary production has no thermostat.
- `run.json` and `traj.jsonl` use every required field from PDF page 10. Save production steps divisible by sample_every, excluding zero. Use sufficient serialization precision for recomputation. Do not overwrite an existing trajectory silently.
- `check` validates metadata, frame count, steps/times, finite numbers, array lengths, wrapped coordinates, and energies recomputed independently from positions/velocities using the naive force path. It rejects malformed or inconsistent files and returns nonzero on failed checks.
- Default-run acceptance: early/late 10% mean-energy drift / abs(first saved energy) <2e-3, abs(T_speed-0.5)<0.05, 24 equal-probability speed bins with chi2/22<2. These contract bounds apply to the unheated default experiment, not to ramp data.
- `make reproduce` creates the default run in ignored artifacts/. A video shows positions alongside g(r), one frame per saved frame, under 2 MB. The supplied viewer is reused unchanged.

## Part 5: measured optimization and melting

- Keep `--force naive`; introduce `--force cells` as the default only after a user-run baseline benchmark/profile identifies force calculation as the target.
- Cells have nx=floor(Lx/rc), ny=floor(Ly/rc). Search wrapped and deduplicated neighbouring cells; visit each unordered pair once. Reuse allocations where helpful. Equality tests include perturbed lattices, periodic-boundary neighbours, cutoff distances, and two-cell-wide boxes.
- Before optimization: three timing runs for the supplied unmodified NumPy baseline, Rust debug, and Rust release, plus a samply profile at N=400, eq_steps=200, steps=1000. Do not fabricate any measurements or infer sample percentages from wall time.
- After optimization: repeat the same profile, benchmark naive/cells at N=100,400,1600 with eq_steps=100 and steps=500, three runs per case. Tables report measured medians and ranges; scaling.png uses seconds per step. Required speedup at N=1600 exceeds 2 and grows with N.
- `--ramp-to` defines a linear target from temperature at production step 0 to ramp_to at the final step, rescaling every 50 production steps. Record ramp_to and test the schedule.
- Heating page: N=400, T=0.2 to 1.2, 20000 steps, sample_every=100, 200 frames. Also create separate fixed-temperature N=100 cold/hot videos at 0.2/1.0, each below 2 MB. Compare long-range g(r) rather than animation speed alone.
- Publish the supplied viewer and heating data in docs/ via GitHub Pages. Verify public access and four synchronized views. Attach the student's <=2-minute, one-take narrated recording to a GitHub release and link it beside the Pages URL.

## Process and boundaries

- Parts 1-2 are directly specified implementation. Parts 3-5 design and ordered plans require the student's review before dependent implementation, as requested by PDF pages 7,12,16.
- Proposed repository instruction change: add only the four PDF-requested upstream Superpowers skills under .agents/skills. Existing AGENTS.md, tutor skill, and global instructions remain authoritative. No expansion of permissions is proposed.
- User initially chose manual grey-box commands after Computer Use refused terminal access, then explicitly delegated the verification/timing checklist back to the agent ("你替我完成"). Use the available terminal tools and preserve actual command receipts. Do not represent agent execution as student verification; do not circumvent the Computer Use terminal restriction.
- The supplied baseline is `/Users/joshua/Downloads/week2-sim.py` and has seed 42. Preserve it unchanged, record its provenance, and disclose this difference from the Rust contract seed 2026.
- Existing authenticated GitHub CLI resolves to Tracer-Huang. The GitHub connector resolves to a different account and must not be used for publication. No login credentials belong in files or commits.
- Public repository commits/pushes, Pages, and release delivery are authorized by the user's full-chain request. Manual recordings and benchmark/profile observations remain evidence dependencies; do not mark them complete without actual results.
