# Gegenbauer coordinates and truncated SVD for the backward heat problem

Code and results for the paper

> M. A. Elshokiry and Y. H. Youssri, *Gegenbauer coordinates and truncated SVD for the backward heat problem:
> an implicit degree weight and the role of boundary conditions*, submitted to the Journal of Mathematical Modeling.

The paper studies how the Gegenbauer parameter λ affects truncated SVD (TSVD) reconstruction of the initial
temperature in the one-dimensional heat equation with Dirichlet or Neumann boundary conditions. All risks are
computed with exact (closed-form) formulas. Random numbers are used only in Section 5.6 and in the Monte Carlo
checks, and they use fixed seeds.

## Installation

Python 3.11 or newer:

```
pip install -r requirements.txt
```

The results in `results/` were produced with Python 3.15.2, NumPy 2.5.4, SciPy 1.18.1 and Matplotlib 3.11.2.

## Reproducing the paper

```
python run_all.py
```

This runs every experiment in the order of the paper and then `verify_paper.py`. Each script prints the numbers
that appear in the paper and writes its results to `results/`. The scripts can also be run one at a time, from any
folder.

## Repository layout

```
gegtsvd/                   library
  model.py                 forward model, Legendre and Gegenbauer bases, benchmarks, priors   (Section 2)
  tsvd.py                  TSVD in general coordinates, exact risk formulas (11) and (13)       (Section 3)
  boundary.py              endpoint values, spherical-Bessel coefficients, synthesis tail      (Sections 4.4, 5.5)
  selection.py             empirical Bayes, GCV and the discrepancy principle                   (Section 5.6)
experiments/               one script per part of the paper
results/                   output of the experiments (CSV and JSON)
figures/                   Figures 1-3 (PDF and EPS)
verify_paper.py            independent check of the printed numbers
run_all.py                 runs everything
```

| Paper | Script | Results |
|---|---|---|
| Sec. 2.3: quadrature against the closed form (7) | `experiments/sec2_forward_model.py` | printed |
| Prop. 1, Sec. 4.1 (meaning of the degree weight), Prop. 3 | `experiments/sec4_degree_weight.py` | `sec4_diagonal_slopes.csv`, `sec4_metric.csv` |
| Sec. 5.1, Table 2, Fig. 1(b) | `experiments/sec5_1_retained_rank.py` | `table2_retained_rank.csv` |
| Sec. 5 (choice of benchmarks), Sec. 5.2, Table 3 | `experiments/sec5_2_benchmarks.py` | `table3_benchmarks.csv` |
| Sec. 5.3, Table 4, Fig. 2 | `experiments/sec5_3_prior_study.py` | `table4_prior_study.csv`, `prior_curves.csv` |
| Sec. 5.4, Table 5 | `experiments/sec5_4_endpoint_energy.py` | `table5_endpoint_energy.csv` |
| Sec. 5.5, Fig. 3 | `experiments/sec5_5_synthesis.py` | `sec5_5_synthesis.json`, `sec5_5_prior_hidden.csv` |
| Sec. 5.6, Table 6 | `experiments/sec5_6_data_driven.py` | `table6_data_driven.csv`, `sec5_6_benchmarks.csv` |
| Figures 1-3 | `experiments/make_figures.py` | `figures/` |

`verify_paper.py` does not use the `gegtsvd` package. It rebuilds the forward model from the description in the
paper, recomputes the numbers of Tables 2 to 4 and Section 5.5, and compares them with the printed values. It
also checks the exact risk formulas against Monte Carlo averages. The expected output ends with
`72/72 checks passed`.

## Computational details

* Heat modes with m ≤ J0 = 250; for Neumann data the constant mode m = 0 is included.
* B is computed by Gauss-Legendre quadrature with 1200 nodes on [0, 1], and C_λ with 2N + 8 nodes (exact).
* Sensors: M = 4(N + 1) interior Gauss-Legendre nodes.
* Singular values below 1e-13 σ1 are treated as zero. The relative threshold gives
  r = #{j : σj ≥ max(ε, 1e-13) σ1}.
* Grids:
  * λ ∈ {0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, …, 2.0};
  * s ∈ {−1, −0.9, …, 5} in Section 5.3, and s ∈ {−0.5, −0.4, …, 5} in Table 3 and Section 5.6;
  * every rank from 0 to the number of non-null directions.

## Citation

See `CITATION.cff`.

## License

MIT (see `LICENSE`).
