"""Section 5.5 and Fig. 3: heat-mode synthesis hides a boundary tail (Proposition 4).

1. The example of the paper: sine benchmark (Dirichlet, N = 15), lambda = 1.5, v = 0.1, threshold 0.1, noise seed 42;
   the same settings under Neumann data (cosine benchmark).
2. Expected hidden error at J = 250 as a fraction of the Bayes risk, for oracle-rank estimates under the priors of
   Section 5.3 (N = 15).
Outputs: results/sec5_5_synthesis.json, results/sec5_5_prior_hidden.csv
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gegtsvd.boundary import endpoint_vectors, tail_exact, tail_leading  # noqa: E402
from gegtsvd.model import (benchmark_coefficients, forward_model, gegenbauer_coordinates, prior_variances,  # noqa: E402
                           weight_coordinates)
from gegtsvd.tsvd import bayes_risk_curve, coordinate_svd, threshold_rank, tsvd_map  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def paper_example(case, bc, lam=1.5, v=0.1, eps=0.1, N=15, seed=42):
    mdl = forward_model(bc, N)
    b = benchmark_coefficients(case, mdl["m"])
    a_true, g = mdl["B"] @ b, mdl["G"] @ b
    svd = coordinate_svd(mdl["A"], gegenbauer_coordinates(N, lam))
    r = threshold_rank(svd, eps)
    y = g + np.sqrt(v) * np.random.default_rng(seed).standard_normal(len(g))
    a = tsvd_map(svd, r) @ y
    ev = endpoint_vectors(N)
    poly_err2 = float(b @ b - a_true @ a_true + np.sum((a - a_true) ** 2))
    out = dict(case=case, bc=bc, rank=r, polynomial_error=float(np.sqrt(poly_err2)),
               q0=float(a @ ev["q0"]), q1=float(a @ ev["q1"]), dq0=float(a @ ev["dq0"]), dq1=float(a @ ev["dq1"]),
               tails=[])
    for J in (250, 500, 1000):
        hidden = tail_exact(a, bc, J)
        out["tails"].append(dict(J=J, hidden_exact=hidden, hidden_leading=tail_leading(a, bc, J), J_times_hidden=J * hidden))
    return out


def prior_hidden(J=250, N=15):
    m_tail = np.arange(J + 1, 200_001).astype(float)
    s2, s2_alt = np.sum(1 / m_tail ** 2), np.sum((-1.0) ** m_tail / m_tail ** 2)
    s4, s4_alt = np.sum(1 / m_tail ** 4), np.sum((-1.0) ** m_tail / m_tail ** 4)
    rows = []
    for bc, alpha in itertools.product(("D", "N"), (0.002, 0.01)):
        mdl = forward_model(bc, N, alpha=alpha)
        A, B, G, m = mdl["A"], mdl["B"], mdl["G"], mdl["m"]
        ev = endpoint_vectors(N)
        coords = {"physical": np.eye(N + 1), "raw Legendre": gegenbauer_coordinates(N, 0.5),
                  "raw lambda=1.5": gegenbauer_coordinates(N, 1.5), "raw lambda=0.05": gegenbauer_coordinates(N, 0.05),
                  "weight s=2": weight_coordinates(N, 2.0)}
        svds = {name: coordinate_svd(A, C) for name, C in coords.items()}
        for beta, v in itertools.product((1.0, 2.0, 3.0, 4.0), (1e-4, 1e-3, 1e-2, 5e-2)):
            sig = prior_variances(m, beta)
            cov_y = (G * sig) @ G.T + v * np.eye(G.shape[0])
            for name, svd in svds.items():
                curve = bayes_risk_curve(svd, B, G, sig, v)
                r = int(np.argmin(curve))
                D = tsvd_map(svd, r)
                keys = ("q0", "q1") if bc == "D" else ("dq0", "dq1")
                e0, e1 = D.T @ ev[keys[0]], D.T @ ev[keys[1]]
                E00, E11, E01 = e0 @ cov_y @ e0, e1 @ cov_y @ e1, e0 @ cov_y @ e1
                if bc == "D":
                    hidden = 2 / np.pi ** 2 * ((E00 + E11) * s2 - 2 * E01 * s2_alt)
                else:
                    hidden = 2 / np.pi ** 4 * ((E00 + E11) * s4 - 2 * E01 * s4_alt)
                rows.append(dict(bc=bc, alpha=alpha, beta=beta, v=v, method=name, rank=r,
                                 bayes_risk=float(curve[r]), hidden_J250=float(hidden),
                                 hidden_fraction=float(hidden / curve[r])))
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    examples = [paper_example("Sine", "D"), paper_example("Cosine", "N")]
    (RESULTS / "sec5_5_synthesis.json").write_text(json.dumps(examples, indent=2))
    for e in examples:
        print(f"{e['case']} ({e['bc']}): rank {e['rank']}, q(0) = {e['q0']:+.2f}, q(1) = {e['q1']:+.2f}")
        for t in e["tails"]:
            print(f"   J={t['J']:5d}: hidden {t['hidden_exact']:.4e} (leading term {t['hidden_leading']:.4e}),"
                  f" J * hidden = {t['J_times_hidden']:.2f}")
    e = examples[0]
    print(f"   predicted constant 2(q(0)^2 + q(1)^2)/pi^2 = {2 * (e['q0'] ** 2 + e['q1'] ** 2) / np.pi ** 2:.2f}")

    rows = prior_hidden()
    with open(RESULTS / "sec5_5_prior_hidden.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("\nExpected hidden error at J = 250 as a fraction of the Bayes risk (oracle rank, N = 15)")
    for bc in ("D", "N"):
        for name in ("physical", "raw Legendre", "raw lambda=1.5", "raw lambda=0.05", "weight s=2"):
            f = np.array([r["hidden_fraction"] for r in rows if r["bc"] == bc and r["method"] == name])
            print(f"  {bc} {name:16s} median {np.median(f):.1e}  max {f.max():.1e}")


if __name__ == "__main__":
    main()
