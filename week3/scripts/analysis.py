"""Rebuild all numerical Part 1-4 evidence from original JSONL rows."""

import json
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stats import (
    acf,
    block_se,
    bootstrap_moments,
    fit_peak,
    stable,
    susceptibility,
    tau_int,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence"
AUX = ROOT / "runs" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)
AUX.mkdir(parents=True, exist_ok=True)
TC = 2 / np.log(1 + np.sqrt(2))
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(name):
    plt.tight_layout()
    plt.savefig(OUT / name)
    plt.close()


def load_run(name):
    folder = ROOT / name
    meta = json.loads((folder / "run.json").read_text())
    data = {}
    cache = folder / "columns.npz"
    if (
        cache.exists()
        and cache.stat().st_mtime > (folder / "series.jsonl").stat().st_mtime
    ):
        z = np.load(cache)
        a = z["a"]
    else:
        rows = []
        with open(folder / "series.jsonl") as f:
            for line in f:
                r = json.loads(line)
                rows.append([r["T"], r["M"], r["E"], r.get("cluster_size", 0)])
        a = np.array(rows)
        np.savez(cache, a=a)
    for t in np.unique(a[:, 0]):
        data[round(float(t), 8)] = a[a[:, 0] == t, 1:]
    return meta, data


def merged(update, l):
    if update == "wolff":
        return load_run(f"artifacts/wolff-l{l}")[1]
    d = load_run(f"artifacts/coarse-l{l}")[1]
    d.update(load_run(f"artifacts/window-l{l}")[1])
    return dict(sorted(d.items()))


def window(update, l):
    return load_run(
        f"artifacts/{'window' if update == 'metropolis' else 'wolff'}-l{l}"
    )[1]


def basic():
    _, hot = load_run("runs/T3.0")
    _, hotter = load_run("runs/T3.1")
    a = hot[3.0][:, 1] * 4096
    b = hotter[3.1][:, 1] * 4096
    edges = np.arange(
        np.floor(min(a.min(), b.min()) / 40) * 40, max(a.max(), b.max()) + 80, 40
    )
    x = (edges[1:] + edges[:-1]) / 2
    h1 = np.histogram(a, edges)[0]
    h2 = np.histogram(b, edges)[0]
    keep = (h1 >= 5) & (h2 >= 5)
    ratio = np.log(h2[keep] / h1[keep])
    slope = 1 / 3 - 1 / 3.1
    intercept = np.mean(ratio - slope * x[keep])
    _fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].stairs(h1, edges, label="T=3.0")
    ax[0].stairs(h2, edges, label="T=3.1")
    ax[0].set(xlabel="Total energy", ylabel="Count")
    ax[0].legend()
    ax[1].plot(x[keep], ratio, "o")
    ax[1].plot(
        x[keep], slope * x[keep] + intercept, "--", label=f"Boltzmann slope {slope:.7f}"
    )
    ax[1].set(xlabel="Total energy", ylabel="log[P3.1(E)/P3.0(E)]")
    ax[1].legend()
    save("boltzmann.png")
    d = {l: merged("metropolis", l) for l in [32, 64]}
    report = {}
    _fig, ax = plt.subplots(figsize=(7, 4))
    for l in [32, 64]:
        ts = np.array(list(d[l]))
        m = np.array([abs(v[:, 0]).mean() for v in d[l].values()])
        ax.plot(ts, m, "o-", ms=3, label=f"L={l}")
        report[str(l)] = {"lowest_abs_m": float(m[0])}
    t = np.linspace(1.5, 3.5, 500)
    exact = np.zeros(len(t))
    mask = t < TC
    exact[mask] = (1 - np.sinh(2 / t[mask]) ** -4) ** 0.125
    ax.plot(t, exact, "k--", label="Infinite-lattice exact")
    ax.axvline(TC, color="gray", ls=":")
    ax.set(xlabel="Temperature", ylabel="Mean |M|")
    ax.legend()
    save("magnetization.png")
    _fig, ax = plt.subplots(figsize=(7, 4))
    for l in [32, 64]:
        w = window("metropolis", l)
        ts = np.array(list(w))
        chi = np.array([susceptibility(v[:, 0], l, t) for t, v in w.items()])
        p, c, center, fit_x = fit_peak(ts, chi)
        report[str(l)]["peak"] = p
        ax.plot(ts, chi, "o-", label=f"L={l}, fitted peak={p:.5f}")
        fine = np.linspace(fit_x[0], fit_x[-1], 100)
        ax.plot(fine, np.polyval(c, fine - center), "--")
    estimate = 2 * report["64"]["peak"] - report["32"]["peak"]
    report["Tc"] = estimate
    report["relative_error"] = abs(estimate - TC) / TC
    ax.axvline(TC, color="gray", ls=":")
    ax.set(xlabel="Temperature", ylabel="Absolute-magnetization susceptibility")
    ax.legend()
    save("susceptibility.png")
    (OUT / "peaks.txt").write_text(json.dumps(report, indent=2) + "\n")
    (AUX / "boltzmann-check.json").write_text(
        json.dumps(
            {
                "expected_slope": slope,
                "unweighted_fitted_slope": float(np.polyfit(x[keep], ratio, 1)[0]),
                "dense_bin_max_residual": float(
                    np.max(
                        abs(
                            (ratio - slope * x[keep] - intercept)[
                                (h1[keep] > 20) & (h2[keep] > 20)
                            ]
                        )
                    )
                ),
            },
            indent=2,
        )
    )


def correlations():
    allrows = []
    _fig, ax = plt.subplots(figsize=(7, 4))
    for l in [32, 64]:
        d = merged("metropolis", l)
        ts = []
        taus = []
        for t, v in d.items():
            x = abs(v[:, 0])
            naive = block_se(x, 1)
            blocked = block_se(x, len(x) // 50)
            tau = tau_int(x)
            allrows.append([l, t, x.mean(), naive, blocked, blocked / naive, tau])
            ts.append(t)
            taus.append(tau)
        ax.semilogy(ts, taus, "o-", label=f"L={l}")
    ax.axvline(TC, color="gray", ls=":")
    ax.set(xlabel="Temperature", ylabel="Integrated autocorrelation time (sweeps)")
    ax.legend()
    save("tau.png")
    np.savetxt(
        OUT / "errors.txt",
        allrows,
        header="L T mean_abs_M naive_SE block50_SE ratio tau_int",
        fmt=["%d", "%.4f"] + ["%.8g"] * 5,
        comments="",
    )
    d = merged("metropolis", 64)
    _fig, ax = plt.subplots(figsize=(8, 4))
    for t in [2.3, 3.0]:
        ax.plot(np.arange(1, 2001), abs(d[t][:2000, 0]), label=f"T={t}", lw=0.8)
    ax.set(xlabel="Measurement sweep", ylabel="|M|")
    ax.legend()
    save("trace.png")
    x = abs(d[2.3][:, 0])
    r = acf(x)
    blocks = np.unique(np.geomspace(1, 10000, 35).astype(int))
    _fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.arange(3001), r[:3001])
    axes[0].set(xlabel="Lag (sweeps)", ylabel="ACF of |M|")
    axes[1].semilogx(blocks, [block_se(x, b) for b in blocks], "o-")
    axes[1].set(xlabel="Block length (sweeps)", ylabel="SE of mean |M|")
    save("acf-binning.png")
    tau = tau_int(x)
    timing = json.loads((ROOT / "artifacts/window-l64/timing.json").read_text())
    rate = 13 * 102000 / timing["wall_seconds"]
    needed = len(x) * 2 * tau
    (AUX / "effective-samples.json").write_text(
        json.dumps(
            {
                "n": len(x),
                "tau": tau,
                "effective_samples": len(x) / (2 * tau),
                "sweeps_per_second_including_IO": rate,
                "sweeps_for_current_naive_error": needed,
                "hours_for_measurement": needed / rate / 3600,
                "assumption": "tau and measured sweep throughput remain fixed; excludes extra equilibration",
                "block_lengths": blocks.tolist(),
                "block_counts": (len(x) // blocks).tolist(),
                "block_SE": [block_se(x, b) for b in blocks],
            },
            indent=2,
        )
    )


def bootstrap(update):
    rng = np.random.default_rng(53152026 if update == "metropolis" else 53152027)
    result = {}
    store = {}
    for b in [2000, 4000, 8000]:
        sizes = {}
        peaks = {}
        store[b] = {}
        for l in [32, 64]:
            w = window(update, l)
            ts = np.array(list(w))
            chis = []
            means = []
            for t, v in w.items():
                a, q = bootstrap_moments(v[:, 0], b, 500, rng)
                chis.append(l * l * (q - a * a) / t)
                means.append(a.std(ddof=1))
            chis = np.array(chis).T
            ps = []
            curves = []
            failed = 0
            for row in chis:
                try:
                    p, c, center, x = fit_peak(ts, row)
                    ps.append(p)
                    curves.append((c, center, x))
                except ValueError:
                    ps.append(np.nan)
                    curves.append(None)
                    failed += 1
            peaks[l] = np.array(ps)
            sizes[str(l)] = {
                "mean_abs_M_se": np.array(means).tolist(),
                "peak_se": float(np.nanstd(ps, ddof=1)),
                "failed_fits": failed,
            }
            store[b][l] = {
                "T": ts,
                "chi": np.array([susceptibility(v[:, 0], l, t) for t, v in w.items()]),
                "fits": curves,
                "mean_se": means,
            }
        tc = 2 * peaks[64] - peaks[32]
        sizes["Tc_se"] = float(np.nanstd(tc, ddof=1))
        sizes["valid_joint_fits"] = int(np.isfinite(tc).sum())
        result[str(b)] = sizes
    result["Tc_error_stable"] = stable(
        [result[str(b)]["Tc_se"] for b in [2000, 4000, 8000]]
    )
    result["method"] = (
        "circular moving-block bootstrap; n retained including partial last block; 500 replicates; each temperature/size independently resampled; peak neighborhood reselected each replicate"
    )
    (AUX / f"{update}-bootstrap.json").write_text(json.dumps(result, indent=2))
    if update == "metropolis":
        _fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, b in zip(axes, [2000, 4000, 8000]):
            for l in [32, 64]:
                s = store[b][l]
                line = ax.plot(s["T"], s["chi"], "o-", ms=3, label=f"L={l}")[0]
                central = fit_peak(s["T"], s["chi"])
                xx = np.linspace(central[3][0], central[3][-1], 150)
                yy = [
                    np.polyval(c, xx - center)
                    for fit in s["fits"]
                    if fit is not None
                    for c, center, _ in [fit]
                ]
                ax.fill_between(
                    xx,
                    np.min(yy, axis=0),
                    np.max(yy, axis=0),
                    color=line.get_color(),
                    alpha=0.2,
                )
                ax.plot(
                    xx,
                    np.polyval(central[1], xx - central[2]),
                    "--",
                    color=line.get_color(),
                )
            ax.axvline(TC, color="gray", ls=":")
            ax.set(
                title=f"block={b}; SE(Tc)={result[str(b)]['Tc_se']:.4f}",
                xlabel="Temperature",
            )
            ax.legend()
        axes[0].set_ylabel("Susceptibility; min/max of 500 fitted curves")
        save("chi-bootstrap.png")
    return result, store


def compare():
    m = json.loads((AUX / "metropolis-bootstrap.json").read_text())
    w = json.loads((AUX / "wolff-bootstrap.json").read_text())
    _fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    comparison = {}
    for update, result in [("metropolis", m), ("wolff", w)]:
        d = window(update, 64)
        ts = np.array(list(d))
        means = np.array([abs(v[:, 0]).mean() for v in d.values()])
        se = result["4000"]["64"]["mean_abs_M_se"]
        axes[0].errorbar(ts, means, yerr=se, fmt="o-", ms=3, capsize=2, label=update)
        i = list(d).index(2.3)
        errs = [result[str(b)]["64"]["mean_abs_M_se"][i] for b in [2000, 4000, 8000]]
        comparison[update] = {
            "mean_T2.3": float(means[i]),
            "SE_T2.3": errs,
            "stable": stable(errs),
        }
    peaks = []
    for l in [32, 64]:
        d = window("wolff", l)
        ts = np.array(list(d))
        chi = np.array([susceptibility(v[:, 0], l, t) for t, v in d.items()])
        p, c, center, x = fit_peak(ts, chi)
        peaks.append(p)
        axes[1].plot(ts, chi, "o-", label=f"L={l}, peak={p:.5f}")
        xx = np.linspace(x[0], x[-1], 100)
        axes[1].plot(xx, np.polyval(c, xx - center), "--")
    tc = 2 * peaks[1] - peaks[0]
    comparison["wolff_Tc"] = tc
    comparison["wolff_relative_error"] = abs(tc - TC) / TC
    denom = np.hypot(
        comparison["metropolis"]["SE_T2.3"][1], comparison["wolff"]["SE_T2.3"][1]
    )
    comparison["combined_SE_distance"] = (
        abs(comparison["metropolis"]["mean_T2.3"] - comparison["wolff"]["mean_T2.3"])
        / denom
    )
    comparison["agreement"] = (
        "discrepancy"
        if comparison["combined_SE_distance"] > 3
        else (
            "agreement"
            if all(comparison[u]["stable"] for u in ["metropolis", "wolff"])
            else "agreement provisional"
        )
    )
    axes[0].set(
        xlabel="Temperature", ylabel="Mean |M|", title="L=64; block=4000 sampling SE"
    )
    axes[1].set(
        xlabel="Temperature",
        ylabel="Wolff susceptibility",
        title=f"Extrapolated Tc={tc:.5f}",
    )
    for ax in axes:
        ax.axvline(TC, color="gray", ls=":")
        ax.legend()
    save("magnetization-compare.png")
    _fig, ax = plt.subplots(figsize=(7, 4))
    vals = {}
    for update in ["metropolis", "wolff"]:
        d = window(update, 64)
        ts = np.array(list(d))
        tau = np.array([tau_int(abs(v[:, 0])) for v in d.values()])
        work = (
            np.ones(len(ts))
            if update == "metropolis"
            else np.array([v[:, 2].mean() / 4096 for v in d.values()])
        )
        ax.semilogy(ts, tau * work, "o-", label=update)
        i = list(d).index(2.3)
        vals[update] = float((tau * work)[i])
        comparison[update]["tau_moves_T2.3"] = float(tau[i])
        comparison[update]["work_per_move_T2.3"] = float(work[i])
    comparison["work_ratio_T2.3"] = vals["metropolis"] / vals["wolff"]
    ax.set(
        xlabel="Temperature",
        ylabel="Work-normalized autocorrelation time",
        title="Spin-update work, not wall-clock speedup",
    )
    ax.legend()
    save("tau-compare.png")
    (AUX / "comparison.json").write_text(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    for func in [basic, correlations]:
        func()
        print(func.__name__, flush=True)
    for u in ["metropolis", "wolff"]:
        bootstrap(u)
        print("bootstrap", u, flush=True)
    compare()
