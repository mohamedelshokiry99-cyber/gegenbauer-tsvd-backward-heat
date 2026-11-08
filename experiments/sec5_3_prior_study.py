"""Section 5.3, Table 4 and Fig. 2: exact Bayes risk over 128 prior-averaged configurations.

Configurations: both boundary conditions, N in {15, 30}, alpha in {0.002, 0.01}, beta in {1, 2, 3, 4} and
v in {1e-4, 1e-3, 1e-2, 5e-2}.  For every family the parameter and the rank minimize the Bayes risk (eq. (13)).
Outputs: results/prior_curves.csv (best rank for every parameter), results/table4_prior_study.csv
"""
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gegtsvd.model import forward_model, gegenbauer_coordinates, prior_variances, weight_coordinates  # noqa: E402
from gegtsvd.tsvd import (bayes_linear_risk, bayes_risk_curve, coordinate_svd,  # noqa: E402
                          eigenbasis_cutoff_risk)

RESULTS = Path(__file__).resolve().parents[1] / "results"
DEGREES, ALPHAS, BETAS, NOISES = (15, 30), (0.002, 0.01), (1.0, 2.0, 3.0, 4.0), (1e-4, 1e-3, 1e-2, 5e-2)
S_GRID = np.round(np.arange(-1.0, 5.0001, 0.1), 1)
LAM_GRID = np.round(np.concatenate([[0.005, 0.01, 0.02, 0.03, 0.05, 0.075], np.arange(0.1, 2.0001, 0.1)]), 3)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    curves, summary = [], []
    for bc, N, alpha in itertools.product(("N", "D"), DEGREES, ALPHAS):
        mdl = forward_model(bc, N, alpha=alpha)
        A, B, G, m = mdl["A"], mdl["B"], mdl["G"], mdl["m"]
        svds = {("weighted", s): coordinate_svd(A, weight_coordinates(N, s)) for s in S_GRID}
        svds |= {("raw", lam): coordinate_svd(A, gegenbauer_coordinates(N, lam)) for lam in LAM_GRID}
        for beta, v in itertools.product(BETAS, NOISES):
            sig = prior_variances(m, beta)
            cfg = dict(bc=bc, N=N, alpha=alpha, beta=beta, v=v)
            best = {}
            for (fam, p), svd in svds.items():
                curve = bayes_risk_curve(svd, B, G, sig, v)
                r = int(np.argmin(curve))
                curves.append(dict(**cfg, family=fam, param=float(p), rank=r, risk=float(curve[r])))
                if fam not in best or curve[r] < best[fam][2]:
                    best[fam] = (float(p), r, float(curve[r]))
            get = lambda fam, p: next(c["risk"] for c in curves[::-1] if c["family"] == fam and c["param"] == p
                                      and all(c[k] == cfg[k] for k in cfg))
            summary.append(dict(**cfg, bayes_linear=bayes_linear_risk(B, G, sig, v),
                                best_s=best["weighted"][0], risk_best_s=best["weighted"][2],
                                best_lam=best["raw"][0], risk_best_lam=best["raw"][2],
                                risk_physical=get("weighted", 0.0), risk_raw_legendre=get("raw", 0.5),
                                risk_eigenbasis=eigenbasis_cutoff_risk(G, sig, v)))
        print(f"done: {bc}, N={N}, alpha={alpha}", flush=True)
    write_csv(RESULTS / "prior_curves.csv", curves)
    write_csv(RESULTS / "table4_prior_study.csv", summary)

    print("\nTable 4: Bayes risk / Bayes-linear bound (geometric mean over 64 configurations)")
    for bc in ("N", "D"):
        sel = [r for r in summary if r["bc"] == bc]
        gm = lambda key: float(np.exp(np.mean([np.log(r[key] / r["bayes_linear"]) for r in sel])))
        print(f"  {'Neumann' if bc == 'N' else 'Dirichlet':9s}: raw Gegenbauer {gm('risk_best_lam'):.2f}  "
              f"explicit weight {gm('risk_best_s'):.2f}  physical {gm('risk_physical'):.2f}  "
              f"raw Legendre {gm('risk_raw_legendre'):.2f}  heat eigenbasis {gm('risk_eigenbasis'):.2f}")
    neu = [r for r in summary if r["bc"] == "N"]
    print("  Neumann, median best s by beta: " + ", ".join(
        f"beta={b:g}: {np.median([r['best_s'] for r in neu if r['beta'] == b]):g}" for b in BETAS))
    dir_ = [r for r in summary if r["bc"] == "D"]
    print(f"  Dirichlet: best s in {sorted({r['best_s'] for r in dir_})}, "
          f"best lambda in [{min(r['best_lam'] for r in dir_):g}, {max(r['best_lam'] for r in dir_):g}]")


if __name__ == "__main__":
    main()
