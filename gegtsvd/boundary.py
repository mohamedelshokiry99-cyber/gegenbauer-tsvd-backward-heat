"""Endpoint values of polynomial estimates and the heat-mode synthesis tail (Sections 4.4, 5.4 and 5.5).

For q = sum_n a_n phi_n on [0, 1], integration by parts gives (Proposition 4)
  Dirichlet:  <q, sqrt2 sin(m pi x)> = sqrt2 [q(0) - (-1)^m q(1)] / (m pi) + O(m^-3),
  Neumann:    <q, sqrt2 cos(m pi x)> = sqrt2 [(-1)^m q'(1) - q'(0)] / (m pi)^2 + O(m^-4).
Exact modal coefficients use the spherical-Bessel representation (eq. (7)).
"""
import numpy as np
from scipy.special import spherical_jn

M_SUM = 200_000       # exact tails are summed up to this mode, plus the leading-term remainder beyond it


def endpoint_vectors(N):
    """Vectors e with q(0) = e_q0 . a, q(1) = e_q1 . a, q'(0) = e_dq0 . a, q'(1) = e_dq1 . a."""
    n = np.arange(N + 1)
    s = np.sqrt(2.0 * n + 1.0)
    d1 = s * n * (n + 1)                       # P_n'(1) = n(n+1)/2, and d/dx = 2 d/dxi
    return dict(q0=s * (-1.0) ** n, q1=s, dq0=d1 * (-1.0) ** (n + 1), dq1=d1)


def modal_coefficients(a, bc, m):
    """<q, psi_m> for q = sum_n a_n phi_n, from sqrt(2n+1) e^{iz} i^n j_n(z), z = m pi / 2."""
    z = m * np.pi / 2.0
    tot = np.zeros(len(m), complex)
    for n, an in enumerate(a):
        tot += an * np.sqrt(2 * n + 1) * (1j) ** n * spherical_jn(n, z)
    tot *= np.exp(1j * z)
    return np.sqrt(2.0) * (tot.real if bc == "N" else tot.imag)


def bessel_B(N, bc, m):
    """Closed-form B[n, m] = <phi_n, psi_m> (eq. (7)); used to check the quadrature."""
    out = np.stack([modal_coefficients(np.eye(N + 1)[n], bc, m) for n in range(N + 1)])
    if bc == "N":
        out[:, m == 0] = 0.0
        out[0, m == 0] = 1.0
    return out


def tail_exact(a, bc, J):
    """Hidden synthesis error ||(I - Pi_J) q||^2."""
    m = np.arange(J + 1, M_SUM + 1)
    c = modal_coefficients(a, bc, m)
    ev = endpoint_vectors(len(a) - 1)
    if bc == "D":
        rem = 2 * ((a @ ev["q0"]) ** 2 + (a @ ev["q1"]) ** 2) / (np.pi ** 2 * M_SUM)
    else:
        rem = 2 * ((a @ ev["dq0"]) ** 2 + (a @ ev["dq1"]) ** 2) / (3 * np.pi ** 4 * M_SUM ** 3)
    return float(np.sum(c ** 2) + rem)


def tail_leading(a, bc, J):
    """Leading-term prediction of ||(I - Pi_J) q||^2 from Proposition 4."""
    ev = endpoint_vectors(len(a) - 1)
    m = np.arange(J + 1, M_SUM + 1).astype(float)
    sgn = (-1.0) ** m
    if bc == "D":
        lead = 2 * (a @ ev["q0"] - sgn * (a @ ev["q1"])) ** 2 / (m * np.pi) ** 2
        rem = 2 * ((a @ ev["q0"]) ** 2 + (a @ ev["q1"]) ** 2) / (np.pi ** 2 * M_SUM)
    else:
        lead = 2 * (sgn * (a @ ev["dq1"]) - a @ ev["dq0"]) ** 2 / (m * np.pi) ** 4
        rem = 2 * ((a @ ev["dq0"]) ** 2 + (a @ ev["dq1"]) ** 2) / (3 * np.pi ** 4 * M_SUM ** 3)
    return float(np.sum(lead) + rem)
