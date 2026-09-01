"""Shared geometry helpers for fem-data scripts.

Import from here after inserting the repo root on ``sys.path``::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from shared import inboard_clamp_phis
"""

from __future__ import annotations

import numpy as np


def _curve_from_coil(obj):
    """Return a simsopt curve from a ``Coil`` or curve-like object."""
    return obj.curve if hasattr(obj, "curve") else obj


def _eval_xyz(curve, phis):
    """Evaluate a ``CurveXYZFourier`` at ``phis`` in ``[0, 1)``.

    Uses the packed DOF layout ``[c(0), s(1), c(1), ..., s(order), c(order)]``
    per Cartesian component, matching :func:`opt_utils.ifft_simsopt`.
    """
    dofs = np.asarray(curve.get_dofs(), dtype=float)
    order = int(curve.order)
    k = 2 * order + 1
    theta = 2.0 * np.pi * np.asarray(phis, dtype=float)
    out = np.empty(theta.shape + (3,), dtype=float)
    for i in range(3):
        c = dofs[i * k:(i + 1) * k]
        val = np.full(theta.shape, c[0], dtype=float)
        for m in range(1, order + 1):
            val = val + c[2 * m - 1] * np.sin(m * theta) + c[2 * m] * np.cos(m * theta)
        out[..., i] = val
    return out


def _curve_center(curve):
    """Fourier ``c0`` centre ``[xc(0), yc(0), zc(0)]``."""
    dofs = np.asarray(curve.get_dofs(), dtype=float)
    k = 2 * int(curve.order) + 1
    return dofs[np.array([0, k, 2 * k])]


def _inboard_phi_for_curve(curve, n_samples=512):
    """Clamp angle at the inboard ``z = z_center`` crossing of one curve."""
    center = _curve_center(curve)
    zc = float(center[2])

    qp = np.asarray(getattr(curve, "quadpoints", []), dtype=float)
    if qp.size >= n_samples:
        phis = qp
    else:
        phis = np.linspace(0.0, 1.0, n_samples, endpoint=False)
    xyz = _eval_xyz(curve, phis)
    dz = xyz[:, 2] - zc

    n = phis.size
    crossings = []
    for i in range(n):
        j = (i + 1) % n
        zi, zj = dz[i], dz[j]
        if zi == 0.0:
            crossings.append(float(phis[i] % 1.0))
            continue
        if zi * zj > 0.0:
            continue
        if zj == 0.0:
            # Count the exact sample on the next iteration (or wrap).
            continue
        dphi = phis[j] - phis[i]
        if dphi <= 0.0:
            dphi += 1.0
        t = zi / (zi - zj)
        crossings.append(float((phis[i] + t * dphi) % 1.0))

    # Unique within a sample step so exact zeros are not double-counted.
    uniq = []
    tol = 0.5 / max(n, 1)
    for phi in crossings:
        if not any(min(abs(phi - u), 1.0 - abs(phi - u)) < tol for u in uniq):
            uniq.append(phi)
    crossings = uniq

    if not crossings:
        raise ValueError(
            "Coil curve never crosses z = z_center; cannot place an inboard clamp."
        )

    pts = _eval_xyz(curve, np.asarray(crossings))
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    return float(crossings[int(np.argmin(r))])


def inboard_clamp_phis(base_coils, n_samples=512):
    """One clamp angle per coil at the inboard midplane.

    For each coil the clamp sits on the curve at the same ``z`` as the
    Fourier centre, on the side of smaller cylindrical radius
    ``r = sqrt(x^2 + y^2)``.

    Parameters
    ----------
    base_coils : sequence of Coil or Curve
        Base coils (or their curves) before symmetry expansion.
    n_samples : int
        Dense sample count used when the curve's own quadpoints are coarser.

    Returns
    -------
    ndarray, shape ``(n_coils, 1)``
        Clamp angles in ``[0, 1)``.
    """
    phis = [
        _inboard_phi_for_curve(_curve_from_coil(obj), n_samples=n_samples)
        for obj in base_coils
    ]
    return np.asarray(phis, dtype=float).reshape(-1, 1)
