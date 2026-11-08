"""Section 2.3: the quadrature matrix B against its closed form (eq. (7)).

Prints the largest absolute difference between B computed by Gauss-Legendre quadrature (1200 nodes) and the
spherical-Bessel closed form, for N in {15, 30} and both boundary conditions.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gegtsvd.boundary import bessel_B  # noqa: E402
from gegtsvd.model import forward_model  # noqa: E402


def main():
    worst = 0.0
    for bc, N in (("N", 15), ("N", 30), ("D", 15), ("D", 30)):
        mdl = forward_model(bc, N)
        diff = np.abs(mdl["B"] - bessel_B(N, bc, mdl["m"])).max()
        worst = max(worst, diff)
        print(f"{bc} N={N}: max |B_quadrature - B_closed_form| = {diff:.2e}   (max |B| = {np.abs(mdl['B']).max():.2f})")
    print(f"largest difference over all cases: {worst:.1e}")


if __name__ == "__main__":
    main()
