"""Compare the optional random-start cooling run with the course ascending data."""

import json

import numpy as np
from analysis import AUX, load_run, merged
from stats import block_se


def main():
    down = load_run("runs/descending")[1]
    up = merged("metropolis", 64)
    rows = []
    for t in sorted(set(down) & set(up)):
        a = abs(up[t][:, 0])
        b = abs(down[t][:, 0])
        se = np.hypot(block_se(a, len(a) // 20), block_se(b, len(b) // 20))
        rows.append(
            {
                "T": t,
                "ascending": float(a.mean()),
                "descending": float(b.mean()),
                "combined_SE_distance": float(abs(a.mean() - b.mean()) / se),
                "agreement_provisional": bool(abs(a.mean() - b.mean()) <= 3 * se),
            }
        )
    agreement = [r["T"] for r in rows if r["agreement_provisional"]]
    report = {
        "lowest_agreeing_temperature": min(agreement) if agreement else None,
        "comparison": rows,
        "interpretation": "One seed and 20-block errors: provisional comparison, not proof of equilibrium or absence of hysteresis. The seed used here did not trap a low-temperature stripe.",
    }
    (AUX / "descending.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
