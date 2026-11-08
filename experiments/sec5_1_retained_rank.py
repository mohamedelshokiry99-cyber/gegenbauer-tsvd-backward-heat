"""Section 5.1, Table 2 and Fig. 1(b): retained rank of raw Gegenbauer TSVD and of its diagonal-only diagnostic.

For each benchmark and lambda in {0.1, ..., 1.5}: the relative-threshold rank with C = C_lambda and with
C = diag(C_lambda), and the smallest exact risk over all ranks (v = 0.05) for both.
Output: results/table2_retained_rank.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gegtsvd.model import BENCHMARKS, benchmark_coefficients, forward_model, gegenbauer_coordinates  # noqa: E402
from gegtsvd.tsvd import coordinate_svd, fixed_risk_curve, threshold_rank  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
V = 0.05


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for case, bm in BENCHMARKS.items():
        mdl = forward_model(bm["bc"], bm["N"])
        b = benchmark_coefficients(case, mdl["m"])
        a, g = mdl["B"] @ b, mdl["G"] @ b
        eN2 = bm["norm2"] - float(a @ a)
        for lam in np.round(np.arange(0.1, 1.5001, 0.1), 1):
            C = gegenbauer_coordinates(bm["N"], lam)
            full, diag = coordinate_svd(mdl["A"], C), coordinate_svd(mdl["A"], np.diag(np.diag(C)))
            rows.append(dict(case=case, lam=lam,
                             rank_raw=threshold_rank(full, bm["eps"]), rank_diag_only=threshold_rank(diag, bm["eps"]),
                             best_risk_raw=float(fixed_risk_curve(full, a, g, eN2, V).min()),
                             best_risk_diag_only=float(fixed_risk_curve(diag, a, g, eN2, V).min())))
    with open(RESULTS / "table2_retained_rank.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print("Table 2: retained rank at the benchmark thresholds, raw Gegenbauer / diagonal of C_lambda only")
    for case in BENCHMARKS:
        sel = [r for r in rows if r["case"] == case and r["lam"] <= 0.5]
        print(f"  {case:10s} " + "  ".join(f"lambda={r['lam']:.1f}: {r['rank_raw']}/{r['rank_diag_only']}" for r in sel))
    worst = max(abs(r["best_risk_diag_only"] / r["best_risk_raw"] - 1) for r in rows
                if r["case"] in ("Triangular", "Cosine") and r["lam"] <= 0.5)
    print(f"Neumann benchmarks, lambda <= 1/2: smallest risk of the diagonal model within {100 * worst:.1f}% of the full one")


if __name__ == "__main__":
    main()
