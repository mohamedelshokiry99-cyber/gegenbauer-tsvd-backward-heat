"""TSVD in general coordinates and its exact risk (Section 3 of the paper).

For a coordinate matrix C, the rank-r estimate is a_hat = D_r(C) y with D_r(C) = C V_r Sigma_r^{-1} U_r^T, where
A C = U Sigma V^T.  All risks below are exact: no Monte Carlo sampling is involved.
"""
import numpy as np

NULL_TOL = 1e-13      # singular values below NULL_TOL * sigma_1 are treated as zero


def coordinate_svd(A, C):
    """SVD of A C without its numerically null directions: (U, s, Vt, C)."""
    U, s, Vt = np.linalg.svd(A @ C, full_matrices=False)
    keep = int(np.sum(s >= NULL_TOL * s[0]))
    return U[:, :keep], s[:keep], Vt[:keep], C


def tsvd_map(svd, r):
    """The linear map D_r(C), shape (N + 1, M)."""
    U, s, Vt, C = svd
    if r == 0:
        return np.zeros((C.shape[0], U.shape[0]))
    return C @ Vt[:r].T @ np.diag(1.0 / s[:r]) @ U[:, :r].T


def threshold_rank(svd, eps):
    """Relative-threshold rank #{j : sigma_j >= max(eps, NULL_TOL) sigma_1}."""
    s = svd[1]
    return int(np.sum(s >= max(eps, NULL_TOL) * s[0]))


def fixed_risk(D, a, g, eN2, v):
    """Exact expected squared error of a fixed map for a known initial state (eq. (11))."""
    return eN2 + float(np.sum((a - D @ g) ** 2)) + v * float(np.sum(D ** 2))


def fixed_risk_curve(svd, a, g, eN2, v):
    """Eq. (11) for every rank r = 0..rank, built up one singular triplet at a time."""
    U, s, Vt, C = svd
    bias, frob = a.copy(), 0.0
    out = [eN2 + float(a @ a)]
    for j in range(len(s)):
        col = C @ Vt[j] / s[j]                 # physical-coefficient direction of the j-th triplet
        bias = bias - col * (U[:, j] @ g)
        frob += float(col @ col)               # ||D_r||_F^2 grows by ||col||^2 (U has orthonormal columns)
        out.append(eN2 + float(bias @ bias) + v * frob)
    return np.array(out)


def bayes_risk_curve(svd, B, G, sig, v):
    """Exact prior-averaged risk (eq. (13)) for every rank r = 0..rank; sig holds the prior variances."""
    U, s, Vt, C = svd
    tail = float(sig.sum() - np.sum((B ** 2) @ sig))    # E||f - Pi_N f||^2
    E, frob = B.copy(), 0.0                              # E = B - D_r G
    out = [tail + float(np.sum(E ** 2 * sig))]
    for j in range(len(s)):
        col = C @ Vt[j] / s[j]
        E -= np.outer(col, U[:, j] @ G)
        frob += float(col @ col)
        out.append(tail + float(np.sum(E ** 2 * sig)) + v * frob)
    return np.array(out)


def bayes_linear_map(B, G, sig, v):
    """Posterior mean in P_N, D* = B S G^T (G S G^T + v I)^{-1}."""
    return (B * sig) @ G.T @ np.linalg.inv((G * sig) @ G.T + v * np.eye(G.shape[0]))


def bayes_linear_risk(B, G, sig, v):
    """Bayes risk of the posterior mean: the lower bound for linear estimators with values in P_N."""
    D = bayes_linear_map(B, G, sig, v)
    E = B - D @ G
    return float(sig.sum() - np.sum((B ** 2) @ sig) + np.sum(E ** 2 * sig) + v * np.sum(D ** 2))


def eigenbasis_cutoff_risk(G, sig, v, K_max=60):
    """Smallest Bayes risk of spectral cut-off in the heat eigenbasis (first K modes, least squares)."""
    best = float(sig.sum())
    for K in range(1, min(G.shape[0], K_max) + 1):
        P = np.linalg.pinv(G[:, :K])
        E = -P @ G
        E[:, :K] += np.eye(K)
        best = min(best, float(sig[K:].sum() + np.sum(E ** 2 * sig) + v * np.sum(P ** 2)))
    return best
