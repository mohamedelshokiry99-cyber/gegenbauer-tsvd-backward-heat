"""Independent check of the numbers printed in the paper.

Written from the text of the paper only: it does not import the gegtsvd package. It rebuilds the forward
model and the algorithm of Section 3.3, recomputes the numbers of Tables 2-4 and Section 5.5, compares them
with the printed values, and checks the exact risk formulas (11) and (13) against Monte Carlo averages.

Run:  python verify_paper.py
"""
import itertools
import time

import numpy as np
from numpy.polynomial import legendre as leg
from scipy.special import eval_gegenbauer, spherical_jn

ALPHA, TAU, J0, Q, NULL = 0.002, 1.0, 250, 1200, 1e-13
RESULTS = []


def check(name, ours, paper, tol):
    ok = abs(ours - paper) <= tol
    RESULTS.append((name, ours, paper, ok))


# ----------------------------------------------------------------- forward matrix (Sec. 2.3)
def gl01(n):
    x, w = leg.leggauss(n)
    return (x + 1) / 2, w / 2


def phi(N, x):                       # phi_n(x) = sqrt(2n+1) P_n(2x-1), L = 1
    return leg.legvander(2 * x - 1, N) * np.sqrt(2 * np.arange(N + 1) + 1)


def modes(bc, x, alpha=ALPHA):
    m = np.arange(0, J0 + 1) if bc == "N" else np.arange(1, J0 + 1)
    if bc == "N":
        psi = np.where(m == 0, 1.0, np.sqrt(2) * np.cos(np.pi * np.outer(x, m)))
    else:
        psi = np.sqrt(2) * np.sin(np.pi * np.outer(x, m))
    return psi, m, np.exp(-alpha * (m * np.pi) ** 2 * TAU)


def forward(bc, N, alpha=ALPHA):
    xq, wq = gl01(Q)
    psi_q, m, dec = modes(bc, xq, alpha)
    B = (phi(N, xq) * wq[:, None]).T @ psi_q                 # B_nm = <phi_n, psi_m>
    xs = gl01(4 * (N + 1))[0]                                 # M = 4(N+1) Gauss-Legendre sensors
    G = modes(bc, xs, alpha)[0] * dec                          # G_im = e^{-alpha mu_m tau} psi_m(x_i)
    return dict(A=G @ B.T, B=B, G=G, m=m)


# ----------------------------------------------------------------- coordinates (eq. (5))
def C_gegenbauer(N, lam):
    xq, wq = gl01(2 * N + 8)
    raw = np.stack([eval_gegenbauer(k, lam, 2 * xq - 1) for k in range(N + 1)], 1)
    return np.triu((phi(N, xq) * wq[:, None]).T @ raw)


def C_weight(N, s):
    return np.diag((np.arange(N + 1) + 1.0) ** (-s))


# ----------------------------------------------------------------- SVD, rank, estimate (Sec. 3.3)
def tsvd(A, C):
    U, sv, Vt = np.linalg.svd(A @ C, full_matrices=False)
    keep = int(np.sum(sv >= NULL * sv[0]))                    # numerically null directions dropped
    return U[:, :keep], sv[:keep], Vt[:keep], C


def D_of(svd, r):
    U, sv, Vt, C = svd
    return C @ Vt[:r].T @ np.diag(1 / sv[:r]) @ U[:, :r].T if r else np.zeros((C.shape[0], U.shape[0]))


def threshold_rank(svd, eps):
    return int(np.sum(svd[1] >= eps * svd[1][0]))


# ----------------------------------------------------------------- exact errors (eqs. (11) and (13))
def risk_fixed(D, a, g, eN2, v):
    return eN2 + np.sum((a - D @ g) ** 2) + v * np.sum(D ** 2)


def risk_curve_fixed(svd, a, g, eN2, v):
    U, sv, Vt, C = svd
    E, frob, out = a.copy(), 0.0, [eN2 + a @ a]
    for j in range(len(sv)):
        col = C @ Vt[j] / sv[j]
        E = E - col * (U[:, j] @ g)
        frob += col @ col
        out.append(eN2 + E @ E + v * frob)
    return np.array(out)


def risk_curve_bayes(svd, B, G, S, v):
    U, sv, Vt, C = svd
    tail = S.sum() - np.sum(B ** 2 @ S)
    E, frob, out = B.copy(), 0.0, [tail + np.sum(B ** 2 @ S)]
    for j in range(len(sv)):
        col = C @ Vt[j] / sv[j]
        E = E - np.outer(col, U[:, j] @ G)
        frob += col @ col
        out.append(tail + np.sum(E ** 2 @ S) + v * frob)
    return np.array(out)


def bayes_linear(B, G, S, v):
    D = (B * S) @ G.T @ np.linalg.inv((G * S) @ G.T + v * np.eye(G.shape[0]))
    return S.sum() - np.sum(B ** 2 @ S) + np.sum((B - D @ G) ** 2 @ S) + v * np.sum(D ** 2)


def eigencut(G, S, v):
    best = S.sum()
    for K in range(1, min(G.shape[0], 60) + 1):
        P = np.linalg.pinv(G[:, :K])
        E = -P @ G
        E[:, :K] += np.eye(K)
        best = min(best, S[K:].sum() + np.sum(E ** 2 @ S) + v * np.sum(P ** 2))
    return best


def truth(case, m):
    b = np.zeros(len(m))
    if case == "Triangular":
        b[m == 0] = 0.5
        sel = m % 4 == 2
        b[sel] = -8 * np.sqrt(2) / (m[sel] * np.pi) ** 2
    elif case == "cos5":
        b[m == 5] = 1 / np.sqrt(2)
    else:
        b[m == 2] = 1 / np.sqrt(2)
    return b


CASES = {"Triangular": ("N", 30, 0.01, 1 / 3), "Sine": ("D", 15, 0.1, 0.5), "Cosine": ("N", 15, 0.1, 0.5)}
LAM_TAB = np.round(np.arange(0.1, 1.5001, 0.1), 1)
S_GRID = np.round(np.arange(-1.0, 5.0001, 0.1), 1)
LAM_PRIOR = np.round(np.concatenate([[0.005, 0.01, 0.02, 0.03, 0.05, 0.075], np.arange(0.1, 2.0001, 0.1)]), 3)


def main():
    t0 = time.time()
    # --- Sec 2.3 check: quadrature B vs closed form (eq. (7))
    worst = 0.0
    for bc, N in (("N", 30), ("D", 15), ("N", 15)):
        f = forward(bc, N)
        z = f["m"] * np.pi / 2
        for n in range(N + 1):
            val = np.sqrt(2 * n + 1) * np.exp(1j * z) * (1j) ** n * spherical_jn(n, z) * np.sqrt(2)
            ref = val.real if bc == "N" else val.imag
            if bc == "N":
                ref = np.where(f["m"] == 0, 1.0 if n == 0 else 0.0, ref)
            worst = max(worst, np.abs(f["B"][n] - ref).max())
    check("B quadrature vs Bessel closed form (max abs diff)", worst, 0.0, 1e-12)

    # --- Prop 1: fitted slope at lambda = 0.1, N = 30
    Ci = np.linalg.inv(C_gegenbauer(30, 0.1))
    k = np.arange(31)
    check("Prop 1 slope, lambda=0.1, N=30", np.polyfit(np.log(k[4:]), np.log(np.diag(Ci)[4:]), 1)[0], 0.885, 5e-4)
    from scipy.special import poch
    check("Prop 1 closed-form diagonal (max rel err)",
          np.max(np.abs(np.diag(Ci) / (np.sqrt(2 * k + 1) * poch(0.5, k) / poch(0.1, k)) - 1)), 0.0, 1e-9)

    # --- Table 2: threshold ranks, raw / diagonal only
    table2 = {"Triangular": [(7, 7), (9, 9), (11, 10), (12, 12), (13, 13)],
              "Sine": [(2, 2), (3, 3), (4, 4), (6, 5), (7, 7)],
              "Cosine": [(2, 2), (3, 3), (5, 5), (6, 6), (7, 7)]}
    # --- Table 3: oracle risks at v = 0.05
    table3 = {"Triangular": (0.003489, 0.003489, 0.003773, 0.003470),
              "Sine": (0.005784, 0.014397, 0.005650, 0.005650),
              "Cosine": (0.005326, 0.005628, 0.007623, 0.005296)}
    for case, (bc, N, eps, f2) in CASES.items():
        f = forward(bc, N)
        b = truth(case, f["m"])
        a, g = f["B"] @ b, f["G"] @ b
        eN2 = f2 - a @ a
        for i, lam in enumerate((0.1, 0.2, 0.3, 0.4, 0.5)):
            Cl = C_gegenbauer(N, lam)
            r_raw = threshold_rank(tsvd(f["A"], Cl), eps)
            r_dg = threshold_rank(tsvd(f["A"], np.diag(np.diag(Cl))), eps)
            check(f"Table 2 {case} lambda={lam} raw rank", r_raw, table2[case][i][0], 0)
            check(f"Table 2 {case} lambda={lam} diag rank", r_dg, table2[case][i][1], 0)
        best = lambda mats: min(risk_curve_fixed(tsvd(f["A"], C), a, g, eN2, 0.05).min() for C in mats)
        ours = (best([C_gegenbauer(N, l) for l in LAM_TAB]), best([C_gegenbauer(N, 0.5)]),
                best([np.eye(N + 1)]), best([C_weight(N, s) for s in S_GRID]))
        for lab, o, p in zip(("raw Gegenbauer", "raw Legendre", "physical", "explicit weight"), ours, table3[case]):
            check(f"Table 3 {case} {lab}", o, p, 6e-7)

    # --- cos(2 pi x) vs cos(5 pi x) with the two cosine maps at rank 5
    f = forward("N", 15)
    for case, lam, paper in (("Cosine", 0.2, 0.005326), ("Cosine", 0.5, 0.005628),
                             ("cos5", 0.2, 0.4751), ("cos5", 0.5, 0.4415)):
        b = truth(case, f["m"])
        a, g = f["B"] @ b, f["G"] @ b
        D = D_of(tsvd(f["A"], C_gegenbauer(15, lam)), 5)
        check(f"truth sensitivity {case} lambda={lam}", risk_fixed(D, a, g, 0.5 - a @ a, 0.05), paper,
              6e-7 if paper < 0.01 else 6e-5)

    # --- Monte Carlo check of the exact fixed-map risk (11): cosine, lambda=0.2, r=5
    b = truth("Cosine", f["m"])
    a, g = f["B"] @ b, f["G"] @ b
    D = D_of(tsvd(f["A"], C_gegenbauer(15, 0.2)), 5)
    rng = np.random.default_rng(2026_10_29)
    Y = g[:, None] + np.sqrt(0.05) * rng.standard_normal((len(g), 200_000))
    loss = (0.5 - a @ a) + np.sum((D @ Y - a[:, None]) ** 2, 0)
    exact = risk_fixed(D, a, g, 0.5 - a @ a, 0.05)
    se = loss.std(ddof=1) / np.sqrt(loss.size)
    check("MC eq. (11): |MC mean - exact| / MC std.error", abs(loss.mean() - exact) / se, 0.0, 3.0)

    # --- Monte Carlo check of the exact Bayes risk: Neumann N=15, beta=2, v=1e-2, physical TSVD
    S = (1 + f["m"].astype(float) ** 2) ** (-2.0)
    S = 0.5 * S / S.sum()
    svd = tsvd(f["A"], np.eye(16))
    curve = risk_curve_bayes(svd, f["B"], f["G"], S, 1e-2)
    r = int(np.argmin(curve))
    D = D_of(svd, r)
    bd = rng.standard_normal((100_000, len(S))) * np.sqrt(S)
    Y = f["G"] @ bd.T + 0.1 * rng.standard_normal((f["G"].shape[0], bd.shape[0]))
    A_ = bd @ f["B"].T
    loss = np.sum(bd ** 2, 1) - np.sum(A_ ** 2, 1) + np.sum((Y.T @ D.T - A_) ** 2, 1)
    se = loss.std(ddof=1) / np.sqrt(loss.size)
    check("MC eq. (13): |MC mean - exact| / MC std.error", abs(loss.mean() - curve[r]) / se, 0.0, 3.0)

    # --- Table 4: prior-averaged geometric means (128 configurations)
    ratios = {bc: {k: [] for k in ("raw", "weight", "phys", "leg", "eig")} for bc in ("N", "D")}
    s_star = {("N", be): [] for be in (1, 2, 3, 4)}
    for bc, N, al in itertools.product(("N", "D"), (15, 30), (0.002, 0.01)):
        f = forward(bc, N, al)
        Cl = {l: tsvd(f["A"], C_gegenbauer(N, l)) for l in LAM_PRIOR}
        Cw = {s: tsvd(f["A"], C_weight(N, s)) for s in S_GRID}
        for be, v in itertools.product((1.0, 2.0, 3.0, 4.0), (1e-4, 1e-3, 1e-2, 5e-2)):
            S = (1 + f["m"].astype(float) ** 2) ** (-be)
            S = 0.5 * S / S.sum()
            lin = bayes_linear(f["B"], f["G"], S, v)
            raw = {l: risk_curve_bayes(sv, f["B"], f["G"], S, v).min() for l, sv in Cl.items()}
            wt = {s: risk_curve_bayes(sv, f["B"], f["G"], S, v).min() for s, sv in Cw.items()}
            ratios[bc]["raw"].append(min(raw.values()) / lin)
            ratios[bc]["weight"].append(min(wt.values()) / lin)
            ratios[bc]["phys"].append(wt[0.0] / lin)
            ratios[bc]["leg"].append(raw[0.5] / lin)
            ratios[bc]["eig"].append(eigencut(f["G"], S, v) / lin)
            if bc == "N":
                s_star[("N", int(be))].append(min(wt, key=wt.get))
    gm = lambda x: float(np.exp(np.mean(np.log(x))))
    table4 = {"N": (1.29, 1.41, 1.98, 1.51, 1.15), "D": (2.04, 1.29, 1.29, 5.45, 1.16)}
    for bc in ("N", "D"):
        for key, p in zip(("raw", "weight", "phys", "leg", "eig"), table4[bc]):
            check(f"Table 4 {bc} {key} (geo-mean)", gm(ratios[bc][key]), p, 0.005)
    for be, p in zip((1, 2, 3, 4), (0.0, 0.2, 0.85, 3.15)):
        check(f"Sec 5.3 Neumann median s*, beta={be}", float(np.median(s_star[("N", be)])), p, 1e-9)

    # --- Sec 5.5: heat-mode synthesis example (sine, lambda=1.5, v=0.1, eps=0.1, seed 42)
    f = forward("D", 15)
    b = truth("Sine", f["m"])
    svd = tsvd(f["A"], C_gegenbauer(15, 1.5))
    r = threshold_rank(svd, 0.1)
    y = f["G"] @ b + np.sqrt(0.1) * np.random.default_rng(42).standard_normal(f["A"].shape[0])
    q = D_of(svd, r) @ y
    n = np.arange(16)
    q0, q1 = q @ (np.sqrt(2 * n + 1) * (-1.0) ** n), q @ np.sqrt(2 * n + 1)
    check("Sec 5.5 rank", r, 12, 0)
    check("Sec 5.5 q(0)", q0, -10.16, 0.005)
    check("Sec 5.5 q(1)", q1, 4.95, 0.005)
    for J, p in ((250, 0.1015), (500, 0.0514), (1000, 0.0258)):
        mm = np.arange(J + 1, 200_001)
        z = mm * np.pi / 2
        c = sum(qn * np.sqrt(2 * k_ + 1) * (1j) ** k_ * spherical_jn(k_, z) for k_, qn in enumerate(q))
        c = np.sqrt(2) * (np.exp(1j * z) * c).imag
        hidden = np.sum(c ** 2) + 2 * (q0 ** 2 + q1 ** 2) / (np.pi ** 2 * 200_000)
        check(f"Sec 5.5 hidden part J={J}", hidden, p, 0.00006)
    check("Sec 5.5 predicted constant 2(q0^2+q1^2)/pi^2", 2 * (q0 ** 2 + q1 ** 2) / np.pi ** 2, 25.86, 0.01)

    # --- report
    w = max(len(r[0]) for r in RESULTS)
    for name, ours, paper, ok in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{w}}  ours={ours:.6g}  paper={paper:.6g}")
    print(f"\n{sum(r[3] for r in RESULTS)}/{len(RESULTS)} checks passed in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
