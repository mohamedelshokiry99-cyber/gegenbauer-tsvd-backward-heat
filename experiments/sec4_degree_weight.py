"""Section 4.1 and Proposition 3: the implicit degree weight of raw Gegenbauer coordinates.

1. Proposition 1: closed-form diagonal of C_lambda^{-1} and its log-log slope against 1 - lambda (N = 30).
2. Meaning of the weight: H = C^-T C^-1 with ratio sqrt(H_kk) / (C^-1)_kk (eq. (16)) and the relative size of the
   off-diagonal part of H, for N in {15, 30} and lambda in {0.05, 0.10, ..., 2.00}.
3. Proposition 3: (1 / 2 lambda) C_lambda -> C_T diag(1 / 2 lambda, 1, 1/2, ..., 1/N) as lambda -> 0.
Outputs: results/sec4_diagonal_slopes.csv, results/sec4_metric.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev
from scipy.special import poch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gegtsvd.model import gauss_legendre, gegenbauer_coordinates, legendre_basis  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def chebyshev_coordinates(N):
    """Physical Legendre coefficients of T_k(2x - 1), k = 0..N."""
    xq, wq = gauss_legendre(2 * N + 8)
    T = np.stack([chebyshev.chebval(2 * xq - 1, np.eye(N + 1)[k]) for k in range(N + 1)], axis=1)
    return (legendre_basis(N, xq) * wq[:, None]).T @ T


def main():
    RESULTS.mkdir(exist_ok=True)

    # 1. Proposition 1
    N = 30
    k = np.arange(N + 1)
    slopes = []
    for lam in np.round(np.arange(0.1, 1.5001, 0.1), 1):
        d = np.diag(np.linalg.inv(gegenbauer_coordinates(N, lam)))
        closed = np.sqrt(2 * k + 1) * poch(0.5, k) / poch(lam, k)
        slope = np.polyfit(np.log(k[4:]), np.log(np.abs(d[4:])), 1)[0]
        slopes.append(dict(N=N, lam=lam, fitted_slope=slope, predicted_slope=1 - lam,
                           max_rel_err_closed_form=float(np.max(np.abs(d / closed - 1)))))
    write_csv(RESULTS / "sec4_diagonal_slopes.csv", slopes)
    print("Proposition 1 (N = 30): fitted log-log slope of (C^-1)_kk, k >= 4, against 1 - lambda")
    for r in slopes:
        print(f"  lambda={r['lam']:.1f}: slope {r['fitted_slope']:.3f}  (1 - lambda = {r['predicted_slope']:.1f}),"
              f"  closed form max rel. error {r['max_rel_err_closed_form']:.1e}")

    # 2. Meaning of the weight
    rows = []
    for N, lam in ((N, lam) for N in (15, 30) for lam in np.round(np.arange(0.05, 2.0001, 0.05), 2)):
        Ci = np.linalg.inv(gegenbauer_coordinates(N, lam))
        H = Ci.T @ Ci
        ratio = np.linalg.norm(Ci, axis=0) / np.abs(np.diag(Ci))     # sqrt(H_kk) / (C^-1)_kk
        rows.append(dict(N=N, lam=lam, max_ratio_sqrtHkk_to_diag=float(ratio.max()),
                         offdiag_fraction_of_H=float(np.linalg.norm(H - np.diag(np.diag(H))) / np.linalg.norm(H))))
    write_csv(RESULTS / "sec4_metric.csv", rows)
    get = lambda N, lam, key: next(r[key] for r in rows if r["N"] == N and abs(r["lam"] - lam) < 1e-9)
    print("\nSection 4.1: penalty of a single degree and coupling between degrees")
    print("  max sqrt(H_kk)/(C^-1)_kk for lambda <= 1.2:",
          f"{max(r['max_ratio_sqrtHkk_to_diag'] for r in rows if r['lam'] <= 1.2 + 1e-9):.3f};",
          f"lambda = 1.5: {get(30, 1.5, 'max_ratio_sqrtHkk_to_diag'):.3f};",
          f"lambda = 2: {get(30, 2.0, 'max_ratio_sqrtHkk_to_diag'):.3f}")
    for lam in (0.05, 0.45, 0.5, 0.55, 0.8, 2.0):
        print(f"  off-diagonal fraction of H at lambda={lam:<4}: N=15 {get(15, lam, 'offdiag_fraction_of_H'):.3f}, "
              f"N=30 {get(30, lam, 'offdiag_fraction_of_H'):.3f}")

    # 3. Proposition 3
    N = 15
    target = chebyshev_coordinates(N) @ np.diag([0.0] + [1.0 / k for k in range(1, N + 1)])
    print("\nProposition 3 (N = 15): max |(1/2 lambda) C_lambda - C_T diag(1/(2 lambda), 1, 1/2, ..., 1/N)|")
    for lam in (1e-2, 1e-3, 1e-4):
        lhs = gegenbauer_coordinates(N, lam) / (2 * lam)
        rhs = target.copy()
        rhs[:, 0] = chebyshev_coordinates(N)[:, 0] / (2 * lam)
        print(f"  lambda={lam:.0e}: {np.abs(lhs - rhs).max():.2e}")


if __name__ == "__main__":
    main()
