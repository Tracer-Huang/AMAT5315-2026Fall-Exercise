"""Course-only raw-data and numerical acceptance checks. No research/NN dependencies."""

import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def validate_run(folder, expected=None):
    folder = pathlib.Path(folder)
    meta = json.loads((folder / "run.json").read_text())
    required = {
        "L",
        "update",
        "t_grid",
        "discard",
        "measure",
        "seed",
        "sample_every",
        "time_unit",
    }

    def integer(value, minimum):
        return type(value) is int and value >= minimum

    if (
        not required <= meta.keys()
        or type(meta["sample_every"]) is not int
        or meta["sample_every"] != 1
    ):
        raise ValueError("metadata contract")
    if meta["update"] not in ("metropolis", "wolff"):
        raise ValueError("unknown update")
    if (
        not all(
            integer(meta[k], low)
            for k, low in [("L", 2), ("discard", 0), ("measure", 1), ("seed", 0)]
        )
        or meta["seed"] > 2**64 - 1
    ):
        raise ValueError("invalid integer metadata")
    grid = meta["t_grid"]
    if (
        not isinstance(grid, list)
        or not grid
        or any(
            type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in grid
        )
    ):
        raise ValueError("invalid temperature grid")
    # Standard course runs ascend. Explicit extension metadata permits descending ramps.
    delta = np.diff(grid)
    if len(delta) and not (
        np.all(delta > 0)
        or (meta.get("extension") == "descending" and np.all(delta < 0))
    ):
        raise ValueError("temperature grid order")
    if meta["time_unit"] != (
        "sweep" if meta["update"] == "metropolis" else "cluster_flip"
    ):
        raise ValueError("time unit")
    if expected is not None:
        for key, value in expected.items():
            if key == "t_grid":
                if len(meta[key]) != len(value) or not np.allclose(
                    meta[key], value, rtol=0, atol=1e-12
                ):
                    raise ValueError("fixed protocol temperature mismatch")
            elif meta.get(key) != value:
                raise ValueError(f"fixed protocol mismatch: {key}")
    count = 0
    moments = np.zeros((len(grid), 2))
    with open(folder / "series.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or not {"L", "T", "sweep", "M", "E"} <= row.keys()
            ):
                raise ValueError("missing row fields")
            if any(type(row[k]) not in (int, float) for k in ("M", "E")):
                raise ValueError("observable types")
            ti = count // meta["measure"]
            step = count % meta["measure"] + 1
            if (
                not all(type(row.get(k)) is int for k in ("L", "sweep"))
                or type(row.get("T")) not in (int, float)
                or not math.isfinite(row["T"])
            ):
                raise ValueError("row integer types/temperature")
            if (
                ti >= len(meta["t_grid"])
                or row.get("sweep") != step
                or row.get("L") != meta["L"]
                or abs(row.get("T", -1) - meta["t_grid"][ti]) > 1e-9
            ):
                raise ValueError("temperature/step order")
            if not (
                math.isfinite(row["M"])
                and abs(row["M"]) <= 1
                and math.isfinite(row["E"])
                and abs(row["E"]) <= 2
            ):
                raise ValueError("observable bounds")
            if meta["update"] == "wolff" and (
                type(row.get("cluster_size")) is not int
                or not 1 <= row.get("cluster_size", 0) <= meta["L"] ** 2
            ):
                raise ValueError("cluster size")
            moments[ti, 0] += abs(row["M"])
            moments[ti, 1] += row["M"] ** 2
            count += 1
    if count != len(meta["t_grid"]) * meta["measure"]:
        raise ValueError("missing rows")
    return {
        "rows": count,
        "moments": [
            {
                "T": t,
                "mean_abs_M": float(v[0] / meta["measure"]),
                "mean_M2": float(v[1] / meta["measure"]),
            }
            for t, v in zip(grid, moments, strict=True)
        ],
        "L": meta["L"],
        "temperatures": len(meta["t_grid"]),
        "update": meta["update"],
    }


def normalize_checks(checks):
    if any(type(value) not in (bool, np.bool_) for value in checks.values()):
        raise ValueError("checks must be Boolean, not truthy numeric values")
    return {key: bool(value) for key, value in checks.items()}


def main():
    from analysis import AUX, merged
    from stats import stable

    out = ROOT / "evidence"
    checks = {}
    details = {}
    specs = {}
    for l, seed in [(32, 1042), (64, 42)]:
        for name, update, discard, measure, lo, hi, step in [
            ("coarse", "metropolis", 2000, 5000, 1.5, 3.5, 0.1),
            ("window", "metropolis", 2000, 100000, 2.0, 2.6, 0.05),
            ("wolff", "wolff", 20000, 100000, 2.0, 2.6, 0.05),
        ]:
            specs[f"{name}-l{l}"] = {
                "L": l,
                "seed": seed,
                "update": update,
                "discard": discard,
                "measure": measure,
                "t_grid": np.arange(lo, hi + step / 2, step).tolist(),
            }
    actual = {p.name for p in (ROOT / "artifacts").iterdir() if p.is_dir()}
    checks["exact_six_named_course_runs"] = actual == set(specs)
    for name, spec in specs.items():
        details[name] = validate_run(ROOT / "artifacts" / name, expected=spec)
    checks["all_six_raw_run_contracts"] = len(details) == 6
    for t in [1.8, 3.0]:
        row = np.loadtxt(ROOT / f"runs/T{t}/stdout.tsv", skiprows=1)
        checks[f"part1_T{t}"] = bool(
            0.95 <= row[1] <= 0.965 if t == 1.8 else row[1] < 0.06
        )
    checks["same_seed_identical"] = (ROOT / "runs/a/series.jsonl").read_bytes() == (
        ROOT / "runs/b/series.jsonl"
    ).read_bytes()
    checks["different_seed_differs"] = (ROOT / "runs/a/series.jsonl").read_bytes() != (
        ROOT / "runs/c/series.jsonl"
    ).read_bytes()
    peaks = json.loads((out / "peaks.txt").read_text())
    comp = json.loads((AUX / "comparison.json").read_text())
    recomputed = {}
    for update, prefix in [("metropolis", "window"), ("wolff", "wolff")]:
        fitted = []
        for l in [32, 64]:
            records = details[f"{prefix}-l{l}"]["moments"]
            temps = np.array([v["T"] for v in records])
            chi = np.array(
                [
                    l * l * (v["mean_M2"] - v["mean_abs_M"] ** 2) / v["T"]
                    for v in records
                ]
            )
            i = int(np.argmax(chi))
            if i < 2 or i + 2 >= len(temps):
                raise ValueError("raw peak lies at fit boundary")
            coeff = np.polyfit(temps[i - 2 : i + 3], chi[i - 2 : i + 3], 2)
            if coeff[0] >= 0:
                raise ValueError("raw peak fit convex")
            vertex = float(-coeff[1] / (2 * coeff[0]))
            if not temps[i - 2] <= vertex <= temps[i + 2]:
                raise ValueError("raw peak outside fit support")
            fitted.append(vertex)
        recomputed[update] = 2 * fitted[1] - fitted[0]
    checks["raw_peak_report_consistency"] = (
        abs(recomputed["metropolis"] - peaks["Tc"]) < 1e-8
        and abs(recomputed["wolff"] - comp["wolff_Tc"]) < 1e-8
    )
    exact_tc = 2 / np.log(1 + np.sqrt(2))
    checks["metropolis_Tc_within_2percent"] = (
        abs(recomputed["metropolis"] - exact_tc) / exact_tc < 0.02
    )
    checks["wolff_Tc_within_2percent"] = (
        abs(recomputed["wolff"] - exact_tc) / exact_tc < 0.02
    )
    checks["both_cold_ordered"] = all(
        peaks[str(l)]["lowest_abs_m"] >= 0.9 for l in [32, 64]
    )
    d = merged("metropolis", 64)
    checks["magnetization_reference_shape"] = all(
        abs(abs(d[t][:, 0]).mean() - target) < 0.1
        for t, target in [(2.25, 0.69), (2.3, 0.41), (2.6, 0.08)]
    )
    for l in [32, 64]:
        checks[f"L{l}_fitted_peak"] = (
            abs(peaks[str(l)]["peak"] - (2.33 if l == 32 else 2.31)) < 0.03
        )
    errors = np.loadtxt(out / "errors.txt", skiprows=1)
    checks["critical_error_ratio_at_least_10"] = (
        float(errors[(errors[:, 0] == 64) & (errors[:, 1] == 2.3), 5][0]) >= 10
    )
    for l, ref in [(32, 190), (64, 670)]:
        tau = errors[errors[:, 0] == l, 6].max()
        checks[f"L{l}_tau_peak_reference_factor2"] = bool(ref / 2 <= tau <= ref * 2)
    checks["wolff_reduces_work_hundreds"] = comp["work_ratio_T2.3"] >= 100
    expected = [
        "boltzmann",
        "magnetization",
        "susceptibility",
        "trace",
        "acf-binning",
        "tau",
        "chi-bootstrap",
        "magnetization-compare",
        "tau-compare",
        "viewer-T1.8",
        "viewer-T2.3",
        "viewer-T3.0",
    ]
    checks["all_12_required_figures"] = all(
        (out / (s + ".png")).is_file() and (out / (s + ".png")).stat().st_size > 1000
        for s in expected
    )
    frames = [json.loads(x) for x in (ROOT / "spins.jsonl").read_text().splitlines()]
    checks["recording_contract"] = len(frames) == 410 and all(
        len(f["spins"]) == 4096
        and set(f["spins"]) <= {-1, 1}
        and abs(sum(f["spins"]) / 4096 - f["m"]) < 1e-12
        for f in frames
    )
    checks["recording_below_5MB"] = (ROOT / "spins.jsonl").stat().st_size < 5_000_000
    checks["recording_frame_times"] = all(
        f["L"] == 64
        and abs(f["T"] - (1.5 + (i // 10) * 0.05)) < 1e-10
        and f["sweep"] == (i // 10) * 2200 + 2000 + (i % 10 + 1) * 20
        for i, f in enumerate(frames)
    )
    bootstrap = {
        u: json.loads((AUX / f"{u}-bootstrap.json").read_text())
        for u in ["metropolis", "wolff"]
    }
    checks["Tc_sampling_errors_stable"] = all(
        bootstrap[u]["Tc_error_stable"] for u in bootstrap
    )
    checks["no_failed_peak_bootstraps"] = all(
        bootstrap[u][str(b)]["valid_joint_fits"] == 500
        for u in bootstrap
        for b in [2000, 4000, 8000]
    )
    expected_agreement = (
        "discrepancy"
        if comp["combined_SE_distance"] > 3
        else (
            "agreement"
            if all(stable(comp[u]["SE_T2.3"]) for u in ["metropolis", "wolff"])
            else "agreement provisional"
        )
    )
    checks["agreement_label_matches_uncertainty"] = (
        comp["agreement"] == expected_agreement
    )
    checks["no_unexplained_sampler_discrepancy"] = expected_agreement != "discrepancy"
    checks["extensions_generated"] = (
        (ROOT / "runs/descending/spins.jsonl").is_file()
        and (AUX / "descending.json").is_file()
        and (AUX / "effective-samples.json").is_file()
    )
    checks = normalize_checks(checks)
    report = {
        "checks": checks,
        "numerical_checks_pass": all(checks.values()),
        "Tc_independently_recomputed": recomputed,
        "sampler_agreement": comp["agreement"],
        "sampling_error_limit": "short-chain mean error remains block-length sensitive; report provisional rather than claim all uncertainties resolved",
        "systematic_error_limit": "two-size extrapolation and five-point fit bias excluded from sampling SE",
        "public_viewer_check": "README links are checked separately after push",
    }
    (AUX / "check.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["numerical_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
