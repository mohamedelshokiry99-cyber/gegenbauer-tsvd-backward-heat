"""Section 5.2 and Table 3: the three benchmarks with oracle choice of the parameter and the rank (v = 0.05).

Also: the truth-sensitivity example (cos(2 pi x) against cos(5 pi x), rank 5) and the polynomial approximation
error of the triangular initial state (Section 5, choice of benchmarks).
Output: results/table3_benchmarks.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gegtsvd.model import (BENCHMARKS, benchmark_coefficients, forward_model, gegenbauer_coordinates,  # noqa: E402
                           weight_coordinates)
from gegtsvd.selection import LAM_GRID, S_GRID  # noqa: E402
from gegtsvd.tsvd import coordinate_svd, fixed_risk, fixed_risk_curve, tsvd_map  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
V = 0.05


def best_over(mdl, a, g, eN2, coords):
    """(risk, parameter, rank) of the best map over a family of coordinate matrices {parameter: C}."""
    best = (np.inf, None, None)
    for p, C in coords.items():
        curve = fixed_risk_curve(coordinate_svd(mdl["A"], C), a, g, eN2, V)
        r = int(np.argmin(curve))
        if curve[r] < best[0]:
            best = (float(curve[r]), p, r)
    return best


def triangle_approximation_error(N):
    """||f - Pi_N f|| for the triangular state, by quadrature split at the kink x = 1/2."""
    x, w = legendre.leggauss(400)
    f = lambda t: np.where(t <= 0.5, 2 * t, 2 * (1 - t))
    proj, total = np.zeros(N + 1), 0.0
    for lo, hi in ((0.0, 0.5), (0.5, 1.0)):
        t, wt = (hi - lo) * (x + 1) / 2 + lo, w * (hi - lo) / 2
        V_ = legendre.legvander(2 * t - 1, N) * np.sqrt(2 * np.arange(N + 1) + 1)
        proj += V_.T @ (wt * f(t))
        total += wt @ f(t) ** 2
    return float(np.sqrt(total - proj @ proj))


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    print("Table 3: smallest expected squared error at v = 0.05 (oracle parameter and rank)")
    for case, bm in BENCHMARKS.items():
        N = bm["N"]
        mdl = forward_model(bm["bc"], N)
        b = benchmark_coefficients(case, mdl["m"])
        a, g = mdl["B"] @ b, mdl["G"] @ b
        eN2 = bm["norm2"] - float(a @ a)
        results = {
            "raw Gegenbauer": best_over(mdl, a, g, eN2, {lam: gegenbauer_coordinates(N, lam) for lam in LAM_GRID}),
            "raw Legendre": best_over(mdl, a, g, eN2, {0.5: gegenbauer_coordinates(N, 0.5)}),
            "physical": best_over(mdl, a, g, eN2, {0.0: np.eye(N + 1)}),
            "explicit weight": best_over(mdl, a, g, eN2, {s: weight_coordinates(N, s) for s in S_GRID}),
        }
        for family, (risk, p, r) in results.items():
            rows.append(dict(case=case, family=family, risk=risk, parameter=p, rank=r))
        print(f"  {case:10s} " + "  ".join(f"{fam}: {x[0]:.6f} ({x[1]:g}, {x[2]})" for fam, x in results.items()))
    with open(RESULTS / "table3_benchmarks.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    print("\nTruth sensitivity (Neumann, N = 15, rank 5, v = 0.05)")
    mdl = forward_model("N", 15)
    for case in ("Cosine", "cos5"):
        b = benchmark_coefficients(case, mdl["m"])
        a, g = mdl["B"] @ b, mdl["G"] @ b
        out = []
        for lam in (0.2, 0.5):
            D = tsvd_map(coordinate_svd(mdl["A"], gegenbauer_coordinates(15, lam)), 5)
            out.append(f"lambda={lam}: {fixed_risk(D, a, g, 0.5 - float(a @ a), V):.4f}")
        print(f"  {'cos(2 pi x)' if case == 'Cosine' else 'cos(5 pi x)'}: " + "  ".join(out))

    print("\nTriangular state: ||f - Pi_N f||")
    errs = {N: triangle_approximation_error(N) for N in (15, 30, 60, 120)}
    Ns = np.array(list(errs))
    rates = -np.diff(np.log(list(errs.values()))) / np.diff(np.log(Ns))
    print("  " + "  ".join(f"N={N}: {e:.2e}" for N, e in errs.items()) + f"   observed rates {np.round(rates, 2)}")


if __name__ == "__main__":
    main()
