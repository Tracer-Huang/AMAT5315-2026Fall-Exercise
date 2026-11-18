"""Verify saved runs and regenerate the three local evidence figures."""
from collections import Counter
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads((ROOT / "runs" / path).read_text())


def run(name):
    return load(f"{name}/run.json")


def save(fig, name):
    path = ROOT / "artifacts" / f"{name}.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gates():
    data = run("gadgets")["evidence"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    for axis, (gate, rows) in zip(axes.flat, [*data["gates"].items(), ("AND → NOT → SET1", data["circuit"])]):
        minimum = min(row["energy"] for row in rows)
        assert all(row["allowed"] == (row["energy"] == minimum) for row in rows)
        assert all(row["energy"] >= minimum + 2 for row in rows if not row["allowed"])
        if gate == "AND":
            assert next(r["energy"] for r in rows if r["inputs"] == [1, 1] and r["output"] == 1) == -3
        if gate == "AND → NOT → SET1":
            assert minimum == -5
            assert [r["bits"] for r in rows if r["energy"] == minimum] == [
                [0, 0, 0, 1], [0, 1, 0, 1], [1, 0, 0, 1],
            ]
        labels = [
            "".join(map(str, row["bits"])) if "bits" in row else
            "".join(map(str, [*row["inputs"], row["output"]]))
            for row in rows
        ]
        axis.bar(labels, [r["energy"] for r in rows],
                 color=["#087e80" if r["allowed"] else "#a4aab3" for r in rows])
        axis.set(title=gate, ylabel="Energy")
        axis.tick_params(axis="x", labelrotation=90)
    fig.suptitle("Gate and circuit energies · allowed rows in teal")
    save(fig, "gates")


def factors(bits, m):
    return tuple(sum(b << i for i, b in enumerate(part)) for part in (bits[:m], bits[m:]))


def plot_factoring():
    assert "CircuitSAT" in load_text("pred/path.txt")
    for target, expected in [(15, (3, 5)), (6, (2, 3)), (7, None)]:
        solve = load(f"pred/factor_{target}.solve.json")
        pair = factors(solve["solution"], 2)
        residual = abs(math.prod(pair) - target)
        assert solve["evaluation"] == f"Min({residual})"
        assert (pair == expected and residual == 0) if expected else residual > 0
        print(f"pred {target}: {pair}, residual {residual}")
    ferro = run("count_p0")["summary"]
    assert (ferro["e_min"], int(ferro["g0"])) == (-180, 2)
    count = run("count_f15")["summary"]
    assert (count["variables"], count["e_min"], int(count["g0"])) == (104, -299, 1)
    assert abs(count["log_z"] - 598.3179) < 0.0001
    sampled = run("sample_f15")
    assert sampled["summary"]["e_min"] == count["e_min"]
    assert abs(sampled["summary"]["log_z"] - count["log_z"]) < 1e-9
    bundle = load("pred/factor_15.bundle.json")["target"]["data"]
    samples = sampled["evidence"]["samples"]
    assert len(samples) == 2000
    for row in samples:
        assert len(row["bits"]) == 104 and set(row["bits"]) <= {0, 1}
        spins = [2*b - 1 for b in row["bits"]]
        energy = sum(j*spins[a]*spins[b] for (a,b),j in
                     zip(bundle["graph"]["edges"], bundle["couplings"]))
        energy += sum(h*s for h,s in zip(bundle["fields"], spins))
        assert energy == row["energy"] >= count["e_min"]
    extracted = load("pred/factor_15.extract.json")
    assert factors(extracted["solution"], 2) == (3, 5)
    assert extracted["evaluation"] == "Min(0)"
    assert extracted["intermediate"]["evaluation"] == "Min(-299)"
    assert load("pred/factor_15.evaluate.json")["result"] == "Min(0)"

    hist = Counter(r["energy"] for r in samples)
    floor = count["e_min"]
    p0 = int(count["g0"]) * math.exp(-2*floor - count["log_z"])
    expected = len(samples)*p0
    sigma = math.sqrt(expected*(1-p0))
    assert abs(hist[floor]-expected) < 5*sigma
    print(f"factor 15: ground states {hist[floor]}/2000, expected {expected:.2f}, SD {sigma:.2f}")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for order in ["greedy", "treesa"]:
        widths = [run(f"plan_k{k}_{order}")["summary"]["sc"] for k in range(2, 9)]
        for k in range(2, 9):
            metrics = run(f"plan_k{k}_{order}")["summary"]
            assert metrics["variables"] == 14*k*k + 8*k
            assert math.isfinite(metrics["tc"]) and metrics["tc"] >= metrics["sc"]
        print(order, widths)
        ax.plot(range(2, 9), widths, "o-", label=order)
    ax.set(xlabel="Bits per factor", ylabel="Peak width (log₂ entries)")
    ax.legend()
    energies = sorted(hist)
    bx.bar([str(e) for e in energies], [hist[e] for e in energies], color="#087e80")
    bx.errorbar(0, expected, yerr=sigma, fmt="o", color="#ad4c26", label="Ground-state expectation ±1 SD")
    bx.set(xlabel="Spin-glass energy", ylabel="Samples", yscale="log", title="N = 15, β = 2")
    bx.tick_params(axis="x", labelrotation=45)
    bx.legend(fontsize=8)
    save(fig, "factoring")


def load_text(path):
    return (ROOT / "runs" / path).read_text()


def plot_queens():
    sizes = list(range(4, 11))
    known = [2, 10, 4, 40, 92, 352, 724]
    counts = [run(f"count_nq{n}")["summary"]["count"] for n in sizes]
    assert counts == known
    for n, reference in [(11, 2680), (12, 14200)]:
        path = ROOT / "runs" / f"count_nq{n}" / "run.json"
        if path.exists():
            assert run(f"count_nq{n}")["summary"]["count"] == reference
    board_hist = {}
    for n in (6, 8):
        result = run(f"sample_nq{n}")
        boards = result["evidence"]["boards"]
        assert len(boards) == 200
        assert result["summary"]["count"] == known[n-4]
        for board in boards:
            assert sorted(board) == list(range(n))
            assert len({i+c for i,c in enumerate(board)}) == n
            assert len({i-c for i,c in enumerate(board)}) == n
        board_hist[n] = Counter(map(tuple, boards))
        print(f"N={n}: {len(board_hist[n])} distinct, example {boards[0]}")
    six = sorted(board_hist[6].items())
    assert len(six) == 4
    print("N=6 frequencies:", six)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    ax.plot(sizes, counts, "o-", label="Contraction")
    ax.scatter(sizes, known, marker="+", color="#ad4c26", label="OEIS A000170")
    ax.set(xlabel="Board size", ylabel="Valid boards", yscale="log", xticks=sizes)
    ax.legend()
    bx.bar(["".join(map(str,b)) for b,_ in six], [c for _,c in six], color="#087e80")
    bx.axhline(50, linestyle="--", color="#ad4c26", label="Uniform expectation")
    bx.set(xlabel="Queen columns in row order (zero-based)", ylabel="Samples", title="N = 6")
    bx.legend()
    save(fig, "nqueens")


if __name__ == "__main__":
    plot_gates()
    plot_factoring()
    plot_queens()
