"""Shared optimization utilities for fem-data scripts.

Import from here (or re-export via a folder's ``optimization.py``). Do not
duplicate these helpers in fixed-*/beams-* modules.

Example::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from opt_utils import load_eq, increase_base_curve_order
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.optimize import NonlinearConstraint
from simsopt.mhd import Vmec
from simsopt.mhd.virtual_casing import VirtualCasing
from simsopt.geo import CurveXYZFourier

# ----- Resolutions -----

n_phi = 25        # half fp like in virtual casing convention
n_theta = 50
vc_src_nphi = 40  # half fp like in virtual casing convention
vc_src_ntheta = 80

# Aim for ~this many quadpoints per base coil (order * ppp).
# Callers (e.g. beams-optimization) may override this attribute after import.
TARGET_QUADPOINTS_PER_COIL = 80


def ifft_simsopt(x, order):
    """Fourier coefficients of a periodic sample array in simsopt's
    ``CurveXYZFourier`` dof order ``[c(0), s(1), c(1), ..., s(order), c(order)]``.

    ``x`` is assumed sampled uniformly over the full period with the endpoint
    excluded, i.e. ``x[j] = f(2*pi*j/n)``.  The returned array has length
    ``2*order + 1`` and reproduces ``f`` truncated to ``order`` via
    ``f(theta) = c(0) + sum_m [ s(m) sin(m theta) + c(m) cos(m theta) ]``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    Xf = np.fft.fft(x)
    dofs = np.zeros(2 * order + 1)
    dofs[0] = Xf[0].real / n                       # cos(0) = mean
    for m in range(1, order + 1):
        # Nyquist mode (even n) has no conjugate partner → half weight.
        fac = 1.0 if (n % 2 == 0 and m == n // 2) else 2.0
        dofs[2 * m - 1] = -fac * Xf[m].imag / n    # sin(m)
        dofs[2 * m]     =  fac * Xf[m].real / n    # cos(m)
    return dofs


def ppp_for_target_quadpoints(order, target=None):
    """Points-per-period so a CurveXYZFourier's ``order*ppp`` count ≈ ``target``.

    ``target`` defaults to the *current* value of ``TARGET_QUADPOINTS_PER_COIL``
    on this module (looked up dynamically), so callers that override that
    attribute after import still take effect.
    """
    if target is None:
        target = TARGET_QUADPOINTS_PER_COIL
    return max(1, int(round(target / order)))


def increase_base_curve_order(base_curves, increment, ppp=None):
    """Re-fit each curve in ``base_curves`` at a higher Fourier order.

    Parameters
    ----------
    base_curves : sequence of CurveXYZFourier
    increment : int
        Added to each curve's current order.
    ppp : int or None
        Points per period. ``None`` (default) chooses ``ppp`` so
        ``new_order * ppp ≈ TARGET_QUADPOINTS_PER_COIL``.
    """
    new_order = base_curves[0].order + increment
    if ppp is None:
        ppp = ppp_for_target_quadpoints(new_order)

    new_curves = [CurveXYZFourier(new_order * ppp, new_order) for _ in base_curves]
    for new_curve, old_curve in zip(new_curves, base_curves):
        gamma = old_curve.gamma()
        dofs = [ifft_simsopt(gamma[:, i], new_order) for i in range(3)]
        new_curve.local_x = np.concatenate(dofs)

    return new_curves


def load_eq(file_name):
    """Load a VMEC wout and build a half-fp virtual-casing plasma surface."""
    eq = Vmec(file_name, keep_all_files=True)
    vc = VirtualCasing.from_vmec(
        file_name,
        src_nphi=vc_src_nphi,
        src_ntheta=vc_src_ntheta,
        trgt_nphi=n_phi,
        trgt_ntheta=n_theta,
    )
    # This is a vacuum case!
    Bnormal_plasma = jnp.zeros_like(vc.B_external_normal)
    plasma_surface_vc = type(eq.boundary)(
        nfp=eq.boundary.nfp,
        stellsym=eq.boundary.stellsym,
        mpol=eq.boundary.mpol, ntor=eq.boundary.ntor,
        quadpoints_phi=np.linspace(0, 1/2/eq.boundary.nfp, n_phi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, n_theta, endpoint=False),
    )
    plasma_surface_vc.set_dofs(eq.boundary.get_dofs())
    return eq, Bnormal_plasma, plasma_surface_vc, vc


def optimizable_to_constraints(optimizable, lb, ub, full_dof_names, prob=None):
    """Wrap a simsopt Optimizable's J()/dJ() as a scipy NonlinearConstraint.

    Parameters
    ----------
    optimizable : simsopt.Optimizable
        Object with .J() (scalar) and .dJ() (gradient wrt its own dofs).
    lb, ub : float
        Lower/upper bounds on optimizable.J().
    full_dof_names : list of str
        dof_names of the full outer optimization problem, i.e. whatever
        indexes the x vector passed to scipy.optimize.minimize.
    prob : simsopt.Problem or Optimizable, optional
        Object whose .x setter should be called to sync state from the
        incoming x before evaluating J()/dJ(). If None, defaults to
        ``optimizable`` itself (only correct if optimizable's own dofs
        span the full problem).

    Returns
    -------
    scipy.optimize.NonlinearConstraint
    """
    if prob is None:
        prob = optimizable

    name_to_idx = {name: i for i, name in enumerate(full_dof_names)}
    local_names = optimizable.dof_names
    local_indices = np.array([name_to_idx[name] for name in local_names])

    def fun(x):
        prob.x = x
        return optimizable.J()

    def jac(x):
        prob.x = x
        grad_local = optimizable.dJ()
        grad_full = np.zeros_like(x, dtype=float)
        grad_full[local_indices] = grad_local
        return grad_full

    return NonlinearConstraint(fun=fun, lb=lb, ub=ub, jac=jac)
