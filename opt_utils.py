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
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint
from simsopt._core.optimizable import Optimizable
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


class _Problem(Optimizable):
    """Union node with no local DOFs that depends on all given Optimizables."""

    def __init__(self, *objs):
        Optimizable.__init__(self, depends_on=list(objs))


def build_problem(objective, constraint_specs, linear_constraint_fns=None):
    """Build a scipy-ready optimization problem from simsopt Optimizables.

    Unifies all Optimizable objects into a single simsopt graph so that
    every constraint and the objective share the same DOF vector.  Gradients
    are scattered by DOF name into the full combined space.

    Parameters
    ----------
    objective : Optimizable
        The objective to minimise; must expose ``J()`` (scalar) and
        ``dJ()`` (gradient w.r.t. its own DOFs).
    constraint_specs : list of (Optimizable, float, float)
        Each entry is ``(obj, lb, ub)``.  ``obj.J()`` is constrained to
        ``[lb, ub]`` and ``obj.dJ()`` supplies the constraint Jacobian row.
    linear_constraint_fns : list of callable or None
        Each callable receives ``prob.dof_names`` (the full DOF name list)
        and returns a ``scipy.optimize.LinearConstraint``.  Evaluated before
        the nonlinear constraints in the returned list.

    Returns
    -------
    x0 : np.ndarray
        Starting point in the full combined DOF space.
    fun : callable
        ``fun(x) -> (J, grad)`` where both are in the full DOF space.
    bounds : scipy.optimize.Bounds
        Box bounds from ``prob.bounds`` in the full DOF space.
    constraints : list
        Ready to pass as ``constraints`` to ``scipy.optimize.minimize``.
    """
    constraint_objs = [obj for obj, _, _ in constraint_specs]
    prob = _Problem(objective, *constraint_objs)
    full_names = prob.dof_names
    n = len(prob.x)

    name_to_idx = {name: i for i, name in enumerate(full_names)}
    obj_indices = np.array([name_to_idx[nm] for nm in objective.dof_names])

    def fun(x):
        prob.x = x
        J = objective.J()
        g_local = objective.dJ()
        g_full = np.zeros(n)
        g_full[obj_indices] = g_local
        return J, g_full

    constraints = []
    if linear_constraint_fns:
        for lc_fn in linear_constraint_fns:
            constraints.append(lc_fn(full_names))

    for obj, lb, ub in constraint_specs:
        constraints.append(
            optimizable_to_constraints(obj, lb, ub, full_names, prob=prob)
        )

    lb_arr, ub_arr = prob.bounds
    return prob.x.copy(), fun, Bounds(lb_arr, ub_arr), constraints


def sum_dphis_A(dof_names, keys=("dphis_start_cc", "dphis_end_cc")):
    """Matrix whose rows sum ``dphis`` increments per coil or group.

    ``dof_names`` are simsopt names such as
    ``CoilSupportBeamsSorted1:dphis_start_cc(i,j)``.  Each row of the
    returned array selects one ``(key, coil)`` group.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for j, name in enumerate(dof_names):
        local = name.split(":", 1)[-1]
        key = local.split("(", 1)[0]
        if key not in keys:
            continue
        i_coil = int(local.split("(", 1)[1].split(",", 1)[0])
        groups[(key, i_coil)].append(j)
    n = len(dof_names)
    if not groups:
        return np.zeros((0, n))
    A = np.zeros((len(groups), n))
    for row, idxs in enumerate(groups.values()):
        A[row, idxs] = 1.0
    return A


def sum_dphis_constraint(dof_names, keys=("dphis_start_cc", "dphis_end_cc")):
    """Linear inequalities ``sum_j dphis[i, j] <= 1`` for each coil or group."""
    A = sum_dphis_A(dof_names, keys)
    ub = np.ones(A.shape[0]) if A.shape[0] else np.zeros(0)
    return LinearConstraint(A, -np.inf, ub)


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
    def curve_from_coil(obj):
        return obj.curve if hasattr(obj, "curve") else obj

    def eval_xyz(curve, phis):
        """``CurveXYZFourier`` at ``phis`` in ``[0, 1)``, simsopt dof order."""
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

    def inboard_phi(curve):
        dofs = np.asarray(curve.get_dofs(), dtype=float)
        k = 2 * int(curve.order) + 1
        zc = float(dofs[2 * k])
        qp = np.asarray(getattr(curve, "quadpoints", []), dtype=float)
        phis = qp if qp.size >= n_samples else np.linspace(0.0, 1.0, n_samples, endpoint=False)
        xyz = eval_xyz(curve, phis)
        dz = xyz[:, 2] - zc
        n = phis.size
        crossings = []
        for i in range(n):
            j = (i + 1) % n
            zi, zj = dz[i], dz[j]
            if zi == 0.0:
                crossings.append(float(phis[i] % 1.0))
                continue
            if zi * zj > 0.0 or zj == 0.0:
                continue
            dphi = phis[j] - phis[i]
            if dphi <= 0.0:
                dphi += 1.0
            t = zi / (zi - zj)
            crossings.append(float((phis[i] + t * dphi) % 1.0))
        uniq = []
        tol = 0.5 / max(n, 1)
        for phi in crossings:
            if not any(min(abs(phi - u), 1.0 - abs(phi - u)) < tol for u in uniq):
                uniq.append(phi)
        if not uniq:
            raise ValueError(
                "Coil curve never crosses z = z_center; cannot place an inboard clamp."
            )
        pts = eval_xyz(curve, np.asarray(uniq))
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        return float(uniq[int(np.argmin(r))])

    phis = [inboard_phi(curve_from_coil(obj)) for obj in base_coils]
    return np.asarray(phis, dtype=float).reshape(-1, 1)
