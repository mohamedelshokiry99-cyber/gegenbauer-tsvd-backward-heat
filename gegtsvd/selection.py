"""Data-driven choice of the coordinates and the rank (Section 5.6 of the paper).

Rules that use only the data y, the forward model and the noise variance v:
  EB   empirical Bayes: fit the prior scale and smoothness beta by maximizing the Gaussian marginal likelihood of y,
       then choose the map (family, parameter, rank) with the smallest plug-in Bayes risk.  The Bayes risk of a fixed
       map is linear in the prior variances, so the plug-in risk is exact.
  GCV  generalized cross-validation over the rank (and the parameter).
  DP   discrepancy principle: the smallest rank with ||y - A a_hat||^2 <= tau^2 M v.
"""
import numpy as np

from .model import gegenbauer_coordinates, weight_coordinates
from .tsvd import NULL_TOL

S_GRID = np.round(np.arange(-0.5, 5.0001, 0.1), 1)
LAM_GRID = np.round(np.concatenate([[0.005, 0.01, 0.02, 0.03, 0.05, 0.075], np.arange(0.1, 2.0001, 0.1)]), 3)
BETA_GRID = np.arange(0.5, 6.0001, 0.25)
ENERGY_GRID = np.logspace(-3, 1, 61)       # candidate values of E||f||^2
DP_TAU = 1.1


class CandidateMaps:
    """Every fixed TSVD map of both coordinate families, with the quantities needed by the rules."""

    def __init__(self, mdl, N):
        A, B, G = mdl["A"], mdl["B"], mdl["G"]
        self.family, self.param, self.rank, maps = [], [], [], []
        for fam, grid, make in (("weighted", S_GRID, lambda p: weight_coordinates(N, p)),
                                ("raw", LAM_GRID, lambda p: gegenbauer_coordinates(N, p))):
            for p in grid:
                C = make(p)
                U, s, Vt = np.linalg.svd(A @ C, full_matrices=False)
                rmax = int(np.sum(s >= NULL_TOL * s[0]))
                D = np.zeros((N + 1, A.shape[0]))
                for r in range(rmax + 1):
                    if r:
                        D = D + np.outer(C @ Vt[r - 1] / s[r - 1], U[:, r - 1])
                    self.family.append(fam)
                    self.param.append(float(p))
                    self.rank.append(r)
                    maps.append(D.copy())
        self.D = np.array(maps)                                  # (candidates, N+1, M)
        self.family, self.param, self.rank = map(np.array, (self.family, self.param, self.rank))
        E = B[None] - self.D @ G
        self.W = np.sum(E ** 2, axis=1)                          # Bayes risk = W @ sigma + v * frob + tail
        self.frob = np.sum(self.D ** 2, axis=(1, 2))
        self.H = A[None] @ self.D                                # hat matrices in data space
        self.M = A.shape[0]

    def mask(self, family=None, param=None):
        keep = np.ones(len(self.family), bool)
        if family is not None:
            keep &= self.family == family
        if param is not None:
            keep &= np.isclose(self.param, param)
        return keep


def empirical_bayes_prior(G, m, Y, v):
    """Per-draw maximum marginal-likelihood prior variances, shape (draws, modes)."""
    best_ll = np.full(Y.shape[1], -np.inf)
    sig_hat = np.zeros((Y.shape[1], len(m)))
    for beta in BETA_GRID:
        shape = (1.0 + m.astype(float) ** 2) ** (-beta)
        lam, Q = np.linalg.eigh((G * shape) @ G.T)
        lam = np.clip(lam, 0, None)
        z2 = (Q.T @ Y) ** 2
        for energy in ENERGY_GRID:
            c = energy / shape.sum()
            den = c * lam + v
            ll = -0.5 * (np.sum(np.log(den)) + (1.0 / den) @ z2)
            upd = ll > best_ll
            best_ll[upd] = ll[upd]
            sig_hat[upd] = c * shape
    return sig_hat


def evaluate_rules(mdl, cand, b_draws, v, rng, sig_true=None):
    """Exact loss of every rule on the initial states b_draws (draws x modes), with fresh noise."""
    B, G, m = mdl["B"], mdl["G"], mdl["m"]
    K = b_draws.shape[0]
    a = b_draws @ B.T
    tail = np.sum(b_draws ** 2, 1) - np.sum(a ** 2, 1)
    Y = G @ b_draws.T + np.sqrt(v) * rng.standard_normal((G.shape[0], K))
    est = np.einsum("cnm,mk->ckn", cand.D, Y)
    loss = np.sum((est - a[None]) ** 2, axis=2) + tail[None]     # (candidates, draws)
    res = np.sum((Y[None] - np.einsum("cij,jk->cik", cand.H, Y)) ** 2, axis=1)
    dof = cand.M - cand.rank                                     # tr(I - H) for TSVD projectors
    gcv = res / np.maximum(dof, 1)[:, None] ** 2
    gcv[dof <= 0] = np.inf
    idx = np.arange(K)
    out = {}

    def pick(score, keep):
        return loss[np.argmin(np.where(keep[:, None], score, np.inf), axis=0), idx]

    sig_hat = empirical_bayes_prior(G, m, Y, v)
    plug = cand.W @ sig_hat.T + v * cand.frob[:, None]
    out["EB, any family"] = pick(plug, np.ones(len(cand.family), bool))
    out["EB, raw Gegenbauer"] = pick(plug, cand.mask("raw"))
    out["EB, explicit weight"] = pick(plug, cand.mask("weighted"))
    out["EB, raw Legendre (rank only)"] = pick(plug, cand.mask("raw", 0.5))
    out["GCV, physical"] = pick(gcv, cand.mask("weighted", 0.0))
    out["GCV, raw Gegenbauer"] = pick(gcv, cand.mask("raw"))
    for name, keep in (("DP, physical", cand.mask("weighted", 0.0)), ("DP, raw Legendre", cand.mask("raw", 0.5))):
        ok = (res <= DP_TAU ** 2 * cand.M * v) & keep[:, None]
        first = np.argmin(np.where(ok, cand.rank[:, None], 10 ** 6), axis=0)    # smallest admissible rank
        first[~ok.any(axis=0)] = np.flatnonzero(keep)[np.argmax(cand.rank[keep])]
        out[name] = loss[first, idx]
    if sig_true is not None:                                     # references that know the prior
        risk = cand.W @ sig_true + v * cand.frob
        for name, keep in (("Oracle, raw Gegenbauer", cand.mask("raw")),
                           ("Oracle, explicit weight", cand.mask("weighted"))):
            out[name] = loss[np.flatnonzero(keep)[np.argmin(risk[keep])]]
        D = (B * sig_true) @ G.T @ np.linalg.inv((G * sig_true) @ G.T + v * np.eye(G.shape[0]))
        out["Bayes-linear"] = np.sum((Y.T @ D.T - a) ** 2, 1) + tail
    return out
