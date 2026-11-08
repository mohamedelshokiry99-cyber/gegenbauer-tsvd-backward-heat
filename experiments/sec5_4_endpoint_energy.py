"""Section 5.4 and Table 5: endpoint energy and risk under Dirichlet data.

For the oracle-rank estimate q = Phi D y in each of the 64 Dirichlet configurations of Section 5.3, the expected
endpoint energy E[q(0)^2 + q(1)^2] is exact: E q(x)^2 = phi(x)^T D (G S G^T + v I) D^T phi(x).
Output: results/table5_endpoint_energy.csv
"""
import csv
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gegtsvd.boundary import endpoint_vectors  # noqa: E402
from gegtsvd.model import forward_model, gegenbauer_coordinates, prior_variances  # noqa: E402
from gegtsvd.tsvd import bayes_linear_risk, bayes_risk_curve, coordinate_svd, tsvd_map  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
LAMBDAS = (0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for N, alpha in itertools.product((15, 30), (0.002, 0.01)):
        mdl = forward_model("D", N, alpha=alpha)
        A, B, G, m = mdl["A"], mdl["B"], mdl["G"], mdl["m"]
        ev = endpoint_vectors(N)
        coords = {"physical": np.eye(N + 1)} | {f"lambda={lam:g}": gegenbauer_coordinates(N, lam) for lam in LAMBDAS}
        svds = {name: coordinate_svd(A, C) for name, C in coords.items()}
        for beta, v in itertools.product((1.0, 2.0, 3.0, 4.0), (1e-4, 1e-3, 1e-2, 5e-2)):
            sig = prior_variances(m, beta)
            ref = bayes_linear_risk(B, G, sig, v)
            cov_y = (G * sig) @ G.T + v * np.eye(G.shape[0])
            for name, svd in svds.items():
                curve = bayes_risk_curve(svd, B, G, sig, v)
                r = int(np.argmin(curve))
                D = tsvd_map(svd, r)
                e0, e1 = D.T @ ev["q0"], D.T @ ev["q1"]              # q(0) = e0 . y, q(1) = e1 . y
                rows.append(dict(N=N, alpha=alpha, beta=beta, v=v, method=name, rank=r,
                                 endpoint_energy=float(e0 @ cov_y @ e0 + e1 @ cov_y @ e1),
                                 risk_over_bayes_linear=float(curve[r] / ref)))
    with open(RESULTS / "table5_endpoint_energy.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print("Table 5 (Dirichlet, 64 configurations, oracle rank)")
    for name in ["physical"] + [f"lambda={lam:g}" for lam in LAMBDAS]:
        e = [r["endpoint_energy"] for r in rows if r["method"] == name]
        q = [r["risk_over_bayes_linear"] for r in rows if r["method"] == name]
        ranks = sorted({r["rank"] for r in rows if r["method"] == name})
        print(f"  {name:11s} median E[q(0)^2+q(1)^2] {np.median(e):.2e}   risk/Bayes-linear (geometric mean) "
              f"{np.exp(np.mean(np.log(q))):.2f}" + ("   (oracle rank 0 everywhere)" if ranks == [0] else ""))
    rho = []
    for key in sorted({(r["N"], r["alpha"], r["beta"], r["v"]) for r in rows}):
        sel = [r for r in rows if (r["N"], r["alpha"], r["beta"], r["v"]) == key
               and r["method"] not in ("physical", "lambda=2")]
        rho.append(spearmanr([r["endpoint_energy"] for r in sel], [r["risk_over_bayes_linear"] for r in sel])[0])
    rho = np.array(rho)
    print(f"  Spearman correlation of endpoint energy and risk across lambda in [0.05, 1.5]: "
          f"positive in {np.sum(rho > 0)}/{len(rho)} configurations, median {np.median(rho):.2f}")


if __name__ == "__main__":
    main()
