"""Forward model of the backward heat problem on [0, 1] (Section 2 of the paper).

Conventions:
  heat modes   Dirichlet: psi_m = sqrt(2) sin(m pi x), m = 1..J0
               Neumann:   psi_0 = 1, psi_m = sqrt(2) cos(m pi x), m = 1..J0
  polynomials  phi_n(x) = sqrt(2n+1) P_n(2x - 1), n = 0..N (orthonormal shifted Legendre basis)
  matrices     B[n, m] = <phi_n, psi_m>,  G[i, m] = exp(-alpha (m pi)^2 tau) psi_m(x_i),  A = G B^T
"""
import numpy as np
from numpy.polynomial import legendre
from scipy.special import eval_gegenbauer

J0 = 250              # heat modes with m <= J0 are kept
QUAD_NODES = 1200     # Gauss-Legendre nodes for B


def gauss_legendre(n):
    """Gauss-Legendre nodes and weights on [0, 1]."""
    x, w = legendre.leggauss(n)
    return (x + 1.0) / 2.0, w / 2.0


def mode_indices(bc, J=J0):
    """Mode indices m in M_J: 0..J for Neumann ('N'), 1..J for Dirichlet ('D')."""
    return np.arange(0, J + 1) if bc == "N" else np.arange(1, J + 1)


def heat_modes(bc, x, J=J0):
    """Values psi_m(x) of the orthonormal heat eigenmodes, shape (len(x), number of modes)."""
    m = mode_indices(bc, J)
    if bc == "N":
        psi = np.sqrt(2.0) * np.cos(np.pi * np.outer(x, m))
        psi[:, m == 0] = 1.0
    else:
        psi = np.sqrt(2.0) * np.sin(np.pi * np.outer(x, m))
    return psi, m


def legendre_basis(N, x):
    """Orthonormal shifted Legendre polynomials phi_0..phi_N at x, shape (len(x), N + 1)."""
    return legendre.legvander(2.0 * x - 1.0, N) * np.sqrt(2.0 * np.arange(N + 1) + 1.0)


def default_sensors(N):
    """M = 4(N + 1) interior Gauss-Legendre sensors."""
    return gauss_legendre(4 * (N + 1))[0]


def forward_model(bc, N, alpha=0.002, tau=1.0, sensors=None, J=J0):
    """Matrices of the discrete forward map (eq. (6)): dict with A, B, G, the mode indices m and the sensors x."""
    x = default_sensors(N) if sensors is None else np.asarray(sensors)
    xq, wq = gauss_legendre(QUAD_NODES)
    psi_q, m = heat_modes(bc, xq, J)
    B = (legendre_basis(N, xq) * wq[:, None]).T @ psi_q
    G = heat_modes(bc, x, J)[0] * np.exp(-alpha * (m * np.pi) ** 2 * tau)
    return dict(A=G @ B.T, B=B, G=G, m=m, x=x)


def gegenbauer_coordinates(N, lam):
    """C_lambda (eq. (5)): physical Legendre coefficients of the raw Gegenbauer polynomials C_k^lambda(2x - 1).

    The integrands are polynomials of degree <= 2N, so Gauss-Legendre quadrature with 2N + 8 nodes is exact.
    """
    if abs(lam - 0.5) < 1e-14:
        lam = 0.5
    xq, wq = gauss_legendre(2 * N + 8)
    raw = np.stack([eval_gegenbauer(k, lam, 2.0 * xq - 1.0) for k in range(N + 1)], axis=1)
    return np.triu((legendre_basis(N, xq) * wq[:, None]).T @ raw)


def weight_coordinates(N, s):
    """Explicit degree weight diag((k + 1)^(-s)) (eq. (8)); s = 0 gives physical TSVD."""
    return np.diag((np.arange(N + 1) + 1.0) ** (-s))


# Benchmark problems of Table 1: boundary condition, degree N, relative threshold, ||f||^2.
BENCHMARKS = {
    "Triangular": dict(bc="N", N=30, eps=0.01, norm2=1.0 / 3.0),
    "Sine": dict(bc="D", N=15, eps=0.1, norm2=0.5),
    "Cosine": dict(bc="N", N=15, eps=0.1, norm2=0.5),
}


def benchmark_coefficients(name, m):
    """Heat-mode coefficients b_m = <f, psi_m> of a benchmark initial state (closed form)."""
    b = np.zeros(len(m))
    if name == "Triangular":            # f = 2x on [0, 1/2], 2(1 - x) on [1/2, 1], cosine series
        b[m == 0] = 0.5
        sel = m % 4 == 2
        b[sel] = -8.0 * np.sqrt(2.0) / (m[sel] * np.pi) ** 2
    elif name in ("Sine", "Cosine"):    # sin(2 pi x) or cos(2 pi x): the single mode m = 2
        b[m == 2] = 1.0 / np.sqrt(2.0)
    elif name == "cos5":                # cos(5 pi x), used for the truth-sensitivity example
        b[m == 5] = 1.0 / np.sqrt(2.0)
    else:
        raise ValueError(name)
    return b


def prior_variances(m, beta):
    """Gaussian smoothness prior (eq. (12)): var(b_m) proportional to (1 + m^2)^(-beta), with E||f||^2 = 1/2."""
    s2 = (1.0 + m.astype(float) ** 2) ** (-beta)
    return 0.5 * s2 / s2.sum()
