"""Section 5.6 and Table 6: data-driven choice of the coordinates and the rank.

48 configurations (both boundary conditions, N = 15, alpha in {0.002, 0.01}, beta in {1, 2, 3, 4},
v in {1e-3, 1e-2, 5e-2}); in each, 300 initial states drawn from the prior with independent noise (fixed seeds).
The loss of every draw is exact.  Also: the three benchmarks (v = 0.05, 500 noise draws).
Outputs: results/table6_data_driven.csv, results/sec5_6_benchmarks.csv
"""
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gegtsvd.model import BENCHMARKS, benchmark_coefficients, forward_model, prior_variances  # noqa: E402
from gegtsvd.selection import CandidateMaps, evaluate_rules  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
RULES = ["Oracle, raw Gegenbauer", "Oracle, explicit weight", "EB, any family", "EB, raw Gegenbauer",
         "EB, raw Legendre (rank only)", "GCV, physical", "GCV, raw Gegenbauer", "DP, physical", "DP, raw Legendre"]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def prior_draws(n_draws=300, N=15):
    rows = []
    for bc, alpha in itertools.product(("N", "D"), (0.002, 0.01)):
        mdl = forward_model(bc, N, alpha=alpha)
        cand = CandidateMaps(mdl, N)
        for beta, v in itertools.product((1.0, 2.0, 3.0, 4.0), (1e-3, 1e-2, 5e-2)):
            seed = 20261027 + int(1000 * beta) + int(1e5 * v) + (0 if bc == "N" else 7) + int(1e4 * alpha)
            rng = np.random.default_rng(seed)
            sig = prior_variances(mdl["m"], beta)
            b_draws = rng.standard_normal((n_draws, len(sig))) * np.sqrt(sig)
            out = evaluate_rules(mdl, cand, b_draws, v, rng, sig_true=sig)
            ref = out["Bayes-linear"].mean()
            for rule, loss in out.items():
                rows.append(dict(bc=bc, alpha=alpha, beta=beta, v=v, rule=rule, mean_loss=float(loss.mean()),
                                 mean_over_bayes_linear=float(loss.mean() / ref),
                                 median_over_bayes_linear=float(np.median(loss) / ref),
                                 catastrophic_fraction=float(np.mean(loss > 10 * ref)), seed=seed, draws=n_draws))
        print(f"done: {bc}, alpha={alpha}", flush=True)
    return rows


def benchmarks(n_draws=500, v=0.05):
    """EB against the exact oracle risk for the three benchmarks (initial states that are not prior draws)."""
    rows = []
    for case, bm in BENCHMARKS.items():
        N = bm["N"]
        mdl = forward_model(bm["bc"], N)
        cand = CandidateMaps(mdl, N)
        b = benchmark_coefficients(case, mdl["m"])
        a = mdl["B"] @ b
        exact = np.sum((a[None] - cand.D @ (mdl["G"] @ b)) ** 2, 1) + v * cand.frob + (b @ b - a @ a)
        rng = np.random.default_rng(42_000 + N + (0 if bm["bc"] == "N" else 1))
        out = evaluate_rules(mdl, cand, np.tile(b, (n_draws, 1)), v, rng)
        oracle = float(exact.min())
        rows.append(dict(case=case, oracle_exact_risk=oracle, eb_mean_loss=float(out["EB, any family"].mean()),
                         ratio=float(out["EB, any family"].mean() / oracle)))
    return rows


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = prior_draws()
    write_csv(RESULTS / "table6_data_driven.csv", rows)
    print("\nTable 6: loss / Bayes-linear risk (geometric mean over 24 configurations per boundary condition)")
    for bc in ("N", "D"):
        print(f"  {'Neumann' if bc == 'N' else 'Dirichlet'}")
        for rule in RULES:
            sel = [r for r in rows if r["bc"] == bc and r["rule"] == rule]
            mean = np.array([r["mean_over_bayes_linear"] for r in sel])
            med = np.array([r["median_over_bayes_linear"] for r in sel])
            fail = np.array([r["catastrophic_fraction"] for r in sel])
            mean_txt = f"{np.exp(np.mean(np.log(mean))):6.2f}" if mean.max() < 1e3 else "     -"
            print(f"    {rule:30s} mean {mean_txt}  median {np.exp(np.mean(np.log(med))):5.2f}  "
                  f"catastrophic {100 * fail.mean():5.1f}%")
    print("\nEB (any family) against the best oracle family, per configuration (ratio of mean losses)")
    for bc in ("N", "D"):
        ratios = []
        for key in sorted({(r["alpha"], r["beta"], r["v"]) for r in rows if r["bc"] == bc}):
            get = lambda rule: next(r["mean_loss"] for r in rows if r["bc"] == bc and r["rule"] == rule
                                    and (r["alpha"], r["beta"], r["v"]) == key)
            ratios.append(get("EB, any family") / min(get("Oracle, raw Gegenbauer"), get("Oracle, explicit weight")))
        print(f"  {bc}: median {np.median(ratios):.3f}, max {np.max(ratios):.3f}")

    bench = benchmarks()
    write_csv(RESULTS / "sec5_6_benchmarks.csv", bench)
    print("\nBenchmarks (v = 0.05): mean loss of EB / exact oracle risk")
    for r in bench:
        print(f"  {r['case']:10s} {r['ratio']:.2f}")


if __name__ == "__main__":
    main()
