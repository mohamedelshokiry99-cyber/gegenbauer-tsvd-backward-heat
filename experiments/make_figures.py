"""Figures 1-3 of the paper (vector PDF and EPS, single-column width).

Fig. 1  implicit degree weight against k^(1 - lambda), and retained rank (needs table2_retained_rank.csv)
Fig. 2  Bayes risk / Bayes-linear bound against lambda (needs prior_curves.csv and table4_prior_study.csv)
Fig. 3  hidden synthesis error against J (computed here)
Run sec5_1_retained_rank.py and sec5_3_prior_study.py first.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.special import poch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gegtsvd.boundary import tail_exact, tail_leading  # noqa: E402
from gegtsvd.model import benchmark_coefficients, forward_model, gegenbauer_coordinates  # noqa: E402
from gegtsvd.tsvd import coordinate_svd, threshold_rank, tsvd_map  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
WIDTH = 3.5                     # single-column width (in)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.0,
    "lines.markersize": 3, "axes.linewidth": 0.6, "legend.frameon": False, "savefig.dpi": 300,
    "pdf.fonttype": 42,
})
C1, C2, C3, C4 = "#0072B2", "#D55E00", "#009E73", "#CC79A7"     # colour-blind-safe (Okabe-Ito)


def load(name):
    with open(RESULTS / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    for ext in ("pdf", "eps"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_weight_rank():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 1.55))
    k = np.arange(1, 31)
    for lam, c in zip((0.1, 0.5, 1.0, 1.5), (C1, C2, C3, C4)):
        w = np.sqrt(2 * k + 1) * poch(0.5, k) / poch(lam, k)
        ax1.loglog(k, w / w[0], "o", color=c, ms=1.8)
        ax1.loglog(k, (k / k[0]) ** (1 - lam), "-", color=c, lw=0.6)
        ax1.text(33, w[-1] / w[0], fr"$\lambda={lam:g}$", color=c, fontsize=6, va="center")
    ax1.set_xlim(0.9, 170)
    ax1.set_xlabel("degree $k$")
    ax1.set_ylabel(r"$(C_\lambda^{-1})_{kk}$ (normalized)")
    ax1.set_title(r"(a) weight vs. $k^{1-\lambda}$ (lines)", loc="left")

    rows = [r for r in load("table2_retained_rank.csv") if r["case"] == "Triangular"]
    lam = [float(r["lam"]) for r in rows]
    ax2.plot(lam, [int(r["rank_raw"]) for r in rows], "o-", color=C1, mfc="none", label=r"raw $C_\lambda$")
    ax2.plot(lam, [int(r["rank_diag_only"]) for r in rows], "s--", color=C2, mfc="none", label=r"diag$(C_\lambda)$ only")
    ax2.axvline(0.5, color="0.6", lw=0.5, ls=":")
    ax2.text(0.47, 16.2, "Legendre", fontsize=6, color="0.4", ha="right")
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel(r"retained rank ($\varepsilon=0.01$)")
    ax2.set_title("(b) triangular benchmark", loc="left")
    ax2.legend(loc="lower right", handletextpad=0.3)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_weight_rank")


def fig_boundary_bayes(N=15, alpha=0.002, v=0.01):
    curves, summary = load("prior_curves.csv"), load("table4_prior_study.csv")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 1.6), sharey=True)
    for ax, bc, title in zip(axes, ("N", "D"), ("(a) Neumann", "(b) Dirichlet")):
        for beta, c in zip((2.0, 3.0, 4.0), (C1, C2, C3)):
            key = lambda r: (r["bc"] == bc and int(r["N"]) == N and float(r["alpha"]) == alpha
                             and float(r["beta"]) == beta and float(r["v"]) == v)
            lin = float(next(r["bayes_linear"] for r in summary if key(r)))
            raw = sorted((float(r["param"]), float(r["risk"])) for r in curves if key(r) and r["family"] == "raw")
            phys = float(next(r["risk"] for r in curves if key(r) and r["family"] == "weighted"
                              and float(r["param"]) == 0.0))
            ax.semilogx([p for p, _ in raw], [x / lin for _, x in raw], "-", color=c, label=fr"$\beta={beta:g}$")
            ax.axhline(phys / lin, color=c, ls="--", lw=0.6)
        ax.axvline(0.5, color="0.6", lw=0.5, ls=":")
        ax.set_yscale("log")
        ax.set_ylim(0.9, 30)
        ax.set_xlabel(r"$\lambda$")
        ax.set_title(title, loc="left")
        ax.text(0.47, 20, "Legendre", fontsize=6, color="0.4", ha="right")
    axes[0].set_ylabel("Bayes risk / bound")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="0.3", ls="--", lw=0.6))
    l.append("physical TSVD")
    axes[0].legend(h, l, loc="upper left", handlelength=1.4, bbox_to_anchor=(0.0, 0.93))
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig_boundary_bayes")


def fig_gibbs_tail():
    Js = np.array([50, 100, 200, 400, 800, 1600])
    fig, ax = plt.subplots(figsize=(WIDTH, 1.65))
    lam, v, eps, N = 1.5, 0.1, 0.1, 15              # the example of Section 5.5 (noise seed 42)
    for case, bc, c, mk in (("Sine", "D", C2, "o"), ("Cosine", "N", C1, "s")):
        mdl = forward_model(bc, N)
        b = benchmark_coefficients(case, mdl["m"])
        svd = coordinate_svd(mdl["A"], gegenbauer_coordinates(N, lam))
        y = mdl["G"] @ b + np.sqrt(v) * np.random.default_rng(42).standard_normal(mdl["A"].shape[0])
        a = tsvd_map(svd, threshold_rank(svd, eps)) @ y
        label = "Dirichlet" if bc == "D" else "Neumann"
        ax.loglog(Js, [tail_exact(a, bc, J) for J in Js], mk, color=c, mfc="none", label=f"{label}: exact")
        ax.loglog(Js, [tail_leading(a, bc, J) for J in Js], "-", color=c, lw=0.7)
    ax.set_xlabel("number of synthesis modes $J$")
    ax.set_ylabel(r"$\|(I-\Pi_J)q\|^2$")
    ax.text(130, 0.9, r"Dirichlet: $\propto J^{-1}$", color=C2, fontsize=7)
    ax.text(60, 2e-5, r"Neumann: $\propto J^{-3}$", color=C1, fontsize=7)
    ax.set_ylim(5e-7, 6.0)
    fig.tight_layout()
    save(fig, "fig_gibbs_tail")


if __name__ == "__main__":
    fig_weight_rank()
    fig_boundary_bayes()
    fig_gibbs_tail()
    print("figures written to", FIGURES)
