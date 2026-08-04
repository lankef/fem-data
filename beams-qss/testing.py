# Verification suite for the RMF frame refactor.
#
# The refactor (coil-fem, geo/framed_curve_jax.py) moved frame construction into
# __init__, stored the RMF as a scalar twist angle relative to the closed-form
# centroid frame, and turned rotated_frame_eval into a smooth viewer on that
# single construction.  SupportBeams._frame_at_phi -- a C0 periodic-linear
# interpolant whose kinks caused the W7-X Taylor gap (dJh/FD = 0.98834) -- was
# deleted.
#
# Phases:
#   frame     Reconstruction fidelity + orthonormality + smoothness of
#             rotated_frame_eval.  Small synthetic curves, no FEM, no GPU.
#   pytree    Frame is built exactly once: pytree round-trips, jit and grad must
#             not rebuild it; with_dofs must.  No FEM, no GPU.
#   meshshift Quantifies the accepted mesh change: the TET10 sweep used to get an
#             RMF re-swept on the K=2M grid and now gets the native-M frame
#             interpolated.  Needs the W7-X coils (meshing only, no solve).
#   smooth    The acceptance test -- AD vs one-sided/centered FD on the W7-X
#             geometry chain, all attachment-angle dofs.  Delegates to
#             geom_smoothness.py.  Needs the W7-X coils, no solve.
#
# The end-to-end check (dJh/FD -> 1.0) is a separate job:
#   sbatch jobscript_taylor.sh
#
# Examples:
#   python -u ./testing.py                      # all phases
#   python -u ./testing.py --phase frame pytree # cheap phases only
#   sbatch jobscript_testing.sh

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

import coil_fem  # noqa: F401  (before gmsh; see taylor.py)
from coil_fem.geo import CurveXYZFourierJAX, make_framed_curve
from coil_fem.geo.framed_curve_jax import (
    FramedCurveRMFJAX,
    _rotated_rmf_frame_pure,
)

jax.config.update("jax_enable_x64", True)

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    _results.append((status, name, detail))
    print(f"  [{status}] {name}" + (f"  --  {detail}" if detail else ""), flush=True)
    return ok


def parse_args():
    p = argparse.ArgumentParser(description="RMF frame refactor verification.")
    p.add_argument(
        "--phase", nargs="+", default=["all"],
        choices=("all", "frame", "pytree", "meshshift", "smooth"),
    )
    p.add_argument("--n-coils", type=int, default=5)
    p.add_argument("--max-dofs", type=int, default=0,
                   help="Limit dofs probed in the smooth phase (0 = all).")
    p.add_argument("--eps", type=float, nargs="+",
                   default=[1e-4, 1e-6, 1e-8])
    return p.parse_args()


# ============================================================================
# Synthetic curves
# ============================================================================

def _wavy_curve(N=32, order=3, seed=0):
    """Non-planar closed curve — exercises a genuinely twisting RMF."""
    qp = jnp.linspace(0.0, 1.0, N, endpoint=False)
    d = jnp.asarray(np.random.default_rng(seed).normal(size=3 * (2 * order + 1)) * 0.1)
    d = d.at[1].set(1.0).at[2 * order + 1 + 2].set(1.0)
    return CurveXYZFourierJAX(qp, d, order)


def _planar_circle(N=4, R=1.0):
    """Degenerate case from tests/test_monolithic.py — RMF closes exactly."""
    qp = jnp.linspace(0.0, 1.0, N, endpoint=False)
    d = jnp.array([0.0, R, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, R])
    return CurveXYZFourierJAX(qp, d, order=1)


# ============================================================================
# frame
# ============================================================================

def phase_frame():
    print(f"\n{'#' * 78}\n### frame: reconstruction, orthonormality, smoothness"
          f"\n{'#' * 78}\n", flush=True)

    for label, curve in (("wavy N=32", _wavy_curve()),
                         ("planar circle N=4", _planar_circle())):
        N = curve.quadpoints.shape[0]
        alpha = jnp.sin(2 * jnp.pi * curve.quadpoints) * 0.3
        fc = FramedCurveRMFJAX(curve, alpha)

        # The twist must reproduce the original scan-based frame exactly.  This
        # is the load-bearing check: rotated_frame() now delegates to
        # rotated_frame_eval, so comparing those two would be vacuous.
        t_r, p_r, q_r = _rotated_rmf_frame_pure(
            curve.gamma(), curve.gammadash(), alpha
        )
        t, p, q = fc.rotated_frame()
        err = max(float(jnp.max(jnp.abs(p - p_r))),
                  float(jnp.max(jnp.abs(q - q_r))),
                  float(jnp.max(jnp.abs(t - t_r))))
        check(f"{label}: twist reconstructs the scan frame", err < 1e-12,
              f"max|delta| = {err:.3e}")

        # Off-grid orthonormality is exact by construction (only a scalar angle
        # is interpolated), not merely approximate.
        phi = jnp.linspace(0.0, 1.0, 97, endpoint=False) + 0.0031
        t, p, q = fc.rotated_frame_eval(phi)
        onorm = max(
            float(jnp.max(jnp.abs(jnp.linalg.norm(p, axis=-1) - 1.0))),
            float(jnp.max(jnp.abs(jnp.linalg.norm(q, axis=-1) - 1.0))),
            float(jnp.max(jnp.abs(jnp.sum(p * q, axis=-1)))),
            float(jnp.max(jnp.abs(jnp.sum(p * t, axis=-1)))),
            float(jnp.max(jnp.abs(jnp.sum(q * t, axis=-1)))),
        )
        check(f"{label}: off-grid frame orthonormal", onorm < 1e-13,
              f"max defect = {onorm:.3e}")

        # Scattered, unsorted phi must give the same answer as sorted -- the old
        # re-sweep was order- and density-dependent.
        idx = np.random.default_rng(1).permutation(phi.shape[0])
        _, p_shuf, _ = fc.rotated_frame_eval(phi[idx])
        check(f"{label}: order-independent",
              float(jnp.max(jnp.abs(p_shuf - p[idx]))) < 1e-14)

        # Smoothness: at a quadpoint knot (the regime that broke before), the
        # one-sided errors must both shrink with eps.  A kink pins |AD-bwd|.
        phi0 = float(curve.quadpoints[N // 4])

        def f(x):
            _, pp, qq = fc.rotated_frame_eval(jnp.atleast_1d(x))
            return jnp.sum(pp) + jnp.sum(qq)

        ad = float(jax.grad(f)(phi0))
        ratios = []
        for eps in (1e-4, 1e-6, 1e-8):
            fwd = float((f(phi0 + eps) - f(phi0)) / eps)
            bwd = float((f(phi0) - f(phi0 - eps)) / eps)
            ratios.append((eps, abs(ad - fwd), abs(ad - bwd)))
        scale = max(abs(ad), 1e-30)
        shrinks = ratios[-1][2] < 0.02 * ratios[0][2] or ratios[-1][2] < 1e-9 * scale
        check(f"{label}: no kink at a quadpoint knot", shrinks,
              "|AD-bwd| " + " -> ".join(f"{r[2]:.2e}" for r in ratios))


# ============================================================================
# pytree
# ============================================================================

def phase_pytree():
    print(f"\n{'#' * 78}\n### pytree: the frame is built exactly once"
          f"\n{'#' * 78}\n", flush=True)

    # Instrument _twist_jitted rather than the pure function: the pure function
    # is behind a jax.jit whose XLA cache would hide repeat *executions*, so
    # counting it would measure tracings.  What matters is whether __init__ asks
    # for a rebuild at all, which is exactly one call to _twist_jitted.
    import coil_fem.geo.framed_curve_jax as fcj

    calls = {"n": 0}
    orig = fcj._twist_jitted

    def counting_twist(*a, **kw):
        calls["n"] += 1
        return orig(*a, **kw)

    fcj._twist_jitted = counting_twist
    try:
        curve = _wavy_curve()
        calls["n"] = 0
        fc = FramedCurveRMFJAX(curve)
        check("construction builds the twist once", calls["n"] == 1,
              f"{calls['n']} call(s)")

        # tree_unflatten routes through __init__, so without the twist being a
        # pytree child every transformation would re-run the lax.scan.
        n0 = calls["n"]
        leaves, treedef = jax.tree_util.tree_flatten(fc)
        for _ in range(5):
            fc2 = jax.tree_util.tree_unflatten(treedef, leaves)
        check("pytree round-trip does not rebuild", calls["n"] == n0,
              f"{calls['n'] - n0} extra call(s)")
        check("round-trip preserves the twist array",
              bool(jnp.all(fc2.twist == fc.twist)))

        n0 = calls["n"]

        @jax.jit
        def use(f, phi):
            _, p, _ = f.rotated_frame_eval(phi)
            return jnp.sum(p)

        phi = jnp.linspace(0.0, 1.0, 7, endpoint=False)
        for _ in range(3):
            float(use(fc, phi))
        check("jit over the frame does not rebuild", calls["n"] == n0,
              f"{calls['n'] - n0} extra call(s)")

        n0 = calls["n"]
        jax.grad(lambda d: use(fc.with_dofs(d), phi))(curve.dofs).block_until_ready()
        check("with_dofs rebuilds exactly once per call", calls["n"] - n0 == 1,
              f"{calls['n'] - n0} call(s)")

        # Centroid frames need no stored state at all.
        fcc = make_framed_curve(curve, "centroid")
        check("centroid frame stores no twist", fcc.twist is None)
    finally:
        fcj._twist_jitted = orig


# ============================================================================
# meshshift
# ============================================================================

def phase_meshshift(n_coils):
    print(f"\n{'#' * 78}\n### meshshift: re-swept vs interpolated frame (TET10)"
          f"\n{'#' * 78}\n", flush=True)
    from geom_smoothness import build_geometry_only

    Jstress, _ = build_geometry_only(n_coils, "staggered")
    fem = Jstress.fem

    worst_ang = 0.0
    worst_disp = 0.0
    coarsest_M = None          # coils may carry different quadpoint counts
    for i, mesh in enumerate(fem.meshes):
        fc = mesh.framed_curve
        M = int(fc.curve.quadpoints.shape[0])
        coarsest_M = M if coarsest_M is None else min(coarsest_M, M)
        K = 2 * M                       # TET10 stride
        phi_grid = jnp.linspace(0.0, 1.0, K, endpoint=False)

        # Old behaviour: rebuild the RMF directly on the refined grid.
        resw_curve = type(fc.curve)(phi_grid, fc.curve.dofs, fc.curve.order)
        _, p_old, q_old = _rotated_rmf_frame_pure(
            resw_curve.gamma(), resw_curve.gammadash(), fc.alpha_eval(phi_grid),
        )
        # New behaviour: interpolate the frame built once on the native grid.
        _, p_new, q_new = fc.rotated_frame_eval(phi_grid)

        dots = jnp.clip(jnp.sum(p_old * p_new, axis=-1), -1.0, 1.0)
        ang = float(jnp.max(jnp.arccos(dots)))
        # Translate the angular change into a node displacement using the
        # cross-section half-width, i.e. what the mesh actually sees.
        half = 0.5 * float(getattr(mesh, "w1", 0.2))
        disp = float(jnp.max(jnp.linalg.norm(p_old - p_new, axis=-1))) * half
        worst_ang = max(worst_ang, ang)
        worst_disp = max(worst_disp, disp)
        print(f"  coil {i}: M={M} K={K}  max angle={ang:.3e} rad  "
              f"max node shift={disp:.3e} m", flush=True)

    # The RMF's own periodic-closure residual is O(1/M), so a shift on that
    # scale is within the model's existing accuracy rather than a new error.
    # "O(1/M)" fixes the scaling, not the coefficient; the factor of 2 is the
    # slack that buys, and anything beyond it is a real change worth looking at.
    scale = 1.0 / coarsest_M
    tol = 2.0 * scale
    print(f"\n  worst angular difference : {worst_ang:.6e} rad", flush=True)
    print(f"  worst node displacement  : {worst_disp:.6e} m", flush=True)
    print(f"  RMF closure accuracy 1/M : {scale:.6e}  (tol = 2/M = {tol:.6e})",
          flush=True)
    check("mesh shift on the scale of the RMF's own 1/M accuracy",
          worst_ang < tol, f"{worst_ang:.3e} vs {tol:.3e}")


# ============================================================================
# smooth
# ============================================================================

def phase_smooth(args):
    print(f"\n{'#' * 78}\n### smooth: acceptance test on the W7-X geometry chain"
          f"\n{'#' * 78}\n", flush=True)
    from geom_smoothness import build_geometry_only, PhiPack, phase_fd

    Jstress, cs = build_geometry_only(args.n_coils, "staggered")
    fem = Jstress.fem
    pack = PhiPack(cs.support_dofs, fem.support)
    worst = phase_fd(fem.support, pack, fem.base_curves_jax,
                     args.eps, args.max_dofs)
    check("no eps-independent one-sided error in the geometry chain",
          worst.get("rel", 1.0) < 1e-4,
          f"worst |AD-bwd| = {worst.get('rel', float('nan')):.3e}")


# ============================================================================

def main():
    args = parse_args()
    phases = args.phase
    if "all" in phases:
        phases = ["frame", "pytree", "meshshift", "smooth"]
    print("testing.py -- RMF frame refactor verification", flush=True)
    print(f"  phases  = {phases}", flush=True)
    print(f"  n_coils = {args.n_coils}", flush=True)

    if "frame" in phases:
        phase_frame()
    if "pytree" in phases:
        phase_pytree()
    if "meshshift" in phases:
        phase_meshshift(args.n_coils)
    if "smooth" in phases:
        phase_smooth(args)

    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    print(f"\n{'=' * 78}\n=== SUMMARY: {len(_results) - n_fail} passed, "
          f"{n_fail} failed ===\n{'=' * 78}", flush=True)
    for status, name, detail in _results:
        if status == FAIL:
            print(f"  [{FAIL}] {name}  --  {detail}", flush=True)
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
