# Tier-0 geometry smoothness diagnostics for the grounded-w_a VJP gap.
#
# Tests hypothesis N1: the same attachment angle phi is evaluated through two
# incompatible representations of the coil --
#
#   * beam_geometry uses curve.gamma_eval(phi)      -> exact Fourier, smooth
#     (beam_network.py:1062)
#   * _surface_exit_params uses _frame_at_phi(phi)  -> periodic LINEAR interp
#     on the curve quadpoint grid via jnp.floor, i.e. C0 only, with a kink at
#     every quadpoint (beam_network.py:622)
#
# The frame flows into _xi_surface_exit -> xi_start/xi_end -> L_eff -> beam
# stiffness.  At a knot, AD returns the right-hand chord slope while a centered
# FD returns the mean of the left and right chord slopes.  That difference is
# O(1/N) and does NOT shrink with eps -- exactly the eps-independent plateau
# bias seen in the W7-X Taylor test (dJh/FD = 0.98834 at eps = 1e-5, 1e-6, 1e-7).
#
# No FEM solve, no adjoint, no cuDSS, no GPU.  Runs on CPU in seconds.
#
# Phases:
#   knots   D1  -- do the attachment angles sit exactly on quadpoint knots?
#   fd      D2  -- one-sided vs centered FD triad on the geometry chain, per dof
#   scan    D3  -- sweep one angle across knots; look for slope jumps in L_eff
#   all         -- all of the above (default)
#
# Examples:
#   python -u ./geom_smoothness.py
#   python -u ./geom_smoothness.py --phase knots
#   python -u ./geom_smoothness.py --phase fd --n-coils 5
#   python -u ./geom_smoothness.py --phase scan --beam 0 --save scan.npz

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Import coil_fem before gmsh (see taylor.py: gmsh first can pull the module
# anaconda libstdc++ ahead of the conda env's and break basix).
import coil_fem  # noqa: F401
from coil_fem.simsopt import CoilFEMObjective, CoilSupportBeams
import gmsh  # noqa: F401
from simsopt.configs import get_data
from simsopt.field import Coil
from simsopt.mhd import Vmec

jax.config.update("jax_enable_x64", True)

# Attachment angles are the only dofs that reach _frame_at_phi.
PHI_KEYS = ("phis_start_cc", "phis_end_cc", "phis_start_cf")

# Geometry quantities probed in the fd phase, in dependency order.  The split
# matters: everything above the line comes from gamma_eval (smooth), everything
# below comes from _frame_at_phi (suspect).
SMOOTH_KEYS = ("x_start", "x_end", "gamma3")
SUSPECT_KEYS = ("xi_start", "xi_end", "L_eff")


def parse_args():
    p = argparse.ArgumentParser(
        description="Tier-0 geometry smoothness diagnostics (no FEM solve)."
    )
    p.add_argument(
        "--phase", default="all", choices=("all", "knots", "fd", "scan"),
    )
    p.add_argument("--n-coils", type=int, default=5)
    p.add_argument(
        "--coupling", default="staggered", choices=("staggered", "monolithic"),
        help="staggered avoids cuDSS/GPU; nothing is solved either way.",
    )
    p.add_argument(
        "--eps", type=float, nargs="+", default=[1e-4, 1e-6, 1e-8],
        help="FD steps for the fd phase (multi-eps is the point).",
    )
    p.add_argument(
        "--max-dofs", type=int, default=0,
        help="Limit the number of phi dofs probed in the fd phase (0 = all).",
    )
    p.add_argument(
        "--beam", type=int, default=-1,
        help="Beam index for --phase scan (-1 = auto: the beam most sensitive "
             "to the scanned dof).",
    )
    p.add_argument(
        "--scan-dof", type=int, default=-1,
        help="phi dof index for --phase scan (-1 = auto: worst dof from fd).",
    )
    p.add_argument("--scan-points", type=int, default=401)
    p.add_argument("--scan-knots", type=float, default=2.0,
                   help="Half-width of the scan in knot spacings.")
    p.add_argument("--save", default="", help="Optional .npz path for scan data.")
    return p.parse_args()


# ============================================================================
# Problem construction (geometry only -- mirrors taylor.build_problem)
# ============================================================================

def build_geometry_only(n_coils: int, coupling: str):
    """Build curves + meshes + SupportBeams.  Never solves.

    ``CoilFEM.__init__`` calls ``support.bind_coil_meshes(self.meshes)``
    (coil_fem.py:324), which is what populates the cross-section metadata that
    ``_surface_exit_params`` needs.  That is the only reason a full
    ``CoilFEMObjective`` is constructed here.
    """
    eq = Vmec("../fixed-continuation/wout.nc", keep_all_files=True)
    n_phi, n_theta = 25, 50
    plasma_surface = type(eq.boundary)(
        nfp=eq.boundary.nfp,
        stellsym=eq.boundary.stellsym,
        mpol=eq.boundary.mpol,
        ntor=eq.boundary.ntor,
        quadpoints_phi=np.linspace(0, 1 / 2 / eq.boundary.nfp, n_phi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, n_theta, endpoint=False),
    )
    plasma_surface.set_dofs(eq.boundary.get_dofs())

    coil_per_half_fp = 5
    if not (1 <= n_coils <= coil_per_half_fp):
        raise ValueError(f"--n-coils must be in 1..{coil_per_half_fp}, got {n_coils}")
    curves, currents, axis, nfp, bs = get_data(
        "w7x", coil_order=8, points_per_period=8
    )
    base_curves = curves[:n_coils]
    base_currents = currents[:n_coils]

    opts = json.load(open(
        Path(__file__).resolve().parent.parent / 'beam-options.json'
    ))
    mesh_options = opts['mesh_options']
    material_options = opts['material_options']
    gravity_options = opts['gravity_options']
    physics_options = opts['physics_options']
    fixed_clamp_options = opts['fixed_clamp_options']
    if n_coils == 1:
        beam_options = {**opts['beam_options'], 'n_beam_cc': 1, 'n_beam_cf': 1}
    else:
        beam_options = opts['beam_options']

    # No 'solver' entry: the geometry path never touches a linear solve, and
    # omitting it keeps this runnable on a CPU-only login node.
    problem_options = {}

    base_coils = [Coil(c, I) for c, I in zip(base_curves, base_currents)]
    coil_support = CoilSupportBeams(
        base_coils=base_coils,
        nfp=eq.boundary.nfp,
        stellsym=eq.boundary.stellsym,
        beam_options=beam_options,
        r_beam=opts['r_beam'],
        fixed_clamp_options=fixed_clamp_options,
        fixed_dof_names=tuple(opts['fixed_dof_names']),
    )
    Jstress = CoilFEMObjective(
        coil_support,
        metrics=("l2_von_mises",),
        metric_weights=(1.0,),
        mesh_options=mesh_options,
        material_options=material_options,
        gravity_options=gravity_options,
        problem_options=problem_options,
        physics_options=physics_options,
        coupling=coupling,
    )
    return Jstress, coil_support


# ============================================================================
# phi dof packing
# ============================================================================

class PhiPack:
    """Flat view over the attachment-angle leaves of ``support_dofs``.

    Also records, for each flat entry, which curve's quadpoint grid the angle
    is interpolated on -- needed to evaluate knot alignment in D1.
    """

    def __init__(self, sdofs, support):
        self.sdofs0 = sdofs
        self.shapes = []
        self.keys = []
        self.curve_idx = []   # per flat entry: curve whose grid the angle uses
        self.label = []       # per flat entry: human-readable name
        flat_parts = []
        for key in PHI_KEYS:
            groups = sdofs.get(key, [])
            for g, arr in enumerate(groups):
                arr = jnp.asarray(arr)
                self.keys.append(key)
                self.shapes.append(arr.shape)
                flat_parts.append(arr.ravel())
                # Which curve grid does _frame_at_phi use for this angle?
                # _surface_exit_params: start angles -> _cc_groups[g][0],
                # end angles -> _cc_groups[g][1], cf angles -> coil g.
                if key == "phis_start_cc":
                    ci = support._cc_groups[g][0]
                elif key == "phis_end_cc":
                    ci = support._cc_groups[g][1]
                else:
                    ci = g
                for j in range(int(arr.size)):
                    self.curve_idx.append(int(ci))
                    self.label.append(f"{key}[{g}][{j}]")
        self.flat0 = (
            jnp.concatenate(flat_parts) if flat_parts else jnp.zeros(0)
        )
        self.n = int(self.flat0.size)

    def unpack(self, flat):
        """Rebuild a support_dofs dict from the flat angle vector."""
        out = dict(self.sdofs0)
        per_key = {k: [] for k in PHI_KEYS}
        i = 0
        for key, shape in zip(self.keys, self.shapes):
            size = int(np.prod(shape)) if shape else 1
            per_key[key].append(flat[i:i + size].reshape(shape))
            i += size
        for key in PHI_KEYS:
            if key in out:
                out[key] = per_key[key]
        return out


# ============================================================================
# D1 -- knot alignment
# ============================================================================

def phase_knots(support, pack):
    print(f"\n{'#' * 78}\n### D1: knot alignment of attachment angles\n{'#' * 78}\n",
          flush=True)

    tmpls = getattr(support, "_coil_framed_templates", None)
    if tmpls is None:
        print("  _coil_framed_templates is None -- meshes not bound; "
              "_surface_exit_params would short-circuit to (0, 1, L).", flush=True)
        return {"bound": False}

    Ns = [int(t.curve.quadpoints.shape[0]) for t in tmpls]
    print("  curve quadpoint counts N per coil:", Ns, flush=True)
    print(f"  N % 8 == 0 for all coils: {all(n % 8 == 0 for n in Ns)}"
          "   (default phis are odd multiples of 1/8)", flush=True)
    print(flush=True)
    print(f"  {'dof':>4}  {'name':<24} {'phi':>18} {'N':>5} {'phi*N':>16} "
          f"{'frac':>12}", flush=True)

    flat = np.asarray(pack.flat0)
    fracs = []
    for j in range(pack.n):
        N = Ns[pack.curve_idx[j]]
        x = (float(flat[j]) % 1.0) * N
        frac = x - np.floor(x)
        fracs.append(frac)
        print(f"  {j:>4}  {pack.label[j]:<24} {flat[j]:>18.12f} {N:>5} "
              f"{x:>16.10f} {frac:>12.3e}", flush=True)

    fracs = np.asarray(fracs)
    dist = np.minimum(fracs, 1.0 - fracs)   # distance to nearest knot
    n_on = int(np.sum(dist < 1e-12))
    print(flush=True)
    print(f"  angles exactly on a knot (dist < 1e-12): {n_on} / {pack.n}", flush=True)
    print(f"  min distance to a knot: {float(dist.min()):.6e}", flush=True)
    print(f"  max distance to a knot: {float(dist.max()):.6e}", flush=True)

    print("\n=== D1 VERDICT ===", flush=True)
    if n_on == pack.n:
        print("  ALL attachment angles sit EXACTLY on quadpoint knots.", flush=True)
        print("  N1 is live: AD takes the right-hand chord slope of the "
              "piecewise-linear", flush=True)
        print("  frame interpolant while centered FD takes the mean of both "
              "sides.", flush=True)
    elif n_on > 0:
        print(f"  {n_on} angles are on knots -- N1 applies to those dofs only.",
              flush=True)
    else:
        print("  NO angle is on a knot (min distance "
              f"{float(dist.min()):.3e}).", flush=True)
        print("  N1 in its knot-alignment form is DEAD.  The frame interpolant "
              "is still", flush=True)
        print("  only C0 in phi, but AD and FD would agree away from knots -- "
              "go to the", flush=True)
        print("  fallback branch (per-dof bisection of lambda-dot-R).", flush=True)
    return {"bound": True, "N": Ns, "n_on_knot": n_on, "dist": dist}


# ============================================================================
# D2 -- one-sided vs centered FD triad on the geometry chain
# ============================================================================

def _geom_fn(pack, support, curves_jax):
    def f(flat):
        sd = pack.unpack(flat)
        g = support.beam_geometry(curves_jax, sd)
        return {k: jnp.asarray(g[k]) for k in SMOOTH_KEYS + SUSPECT_KEYS}
    return f


def _rel(a, b):
    """Max-norm relative difference, guarded for a ~zero reference."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    scale = max(float(np.max(np.abs(a))), 1e-300)
    return float(np.max(np.abs(a - b))) / scale


def phase_fd(support, pack, curves_jax, eps_list, max_dofs):
    print(f"\n{'#' * 78}\n### D2: one-sided vs centered FD on the geometry chain"
          f"\n{'#' * 78}\n", flush=True)
    print("  At a SMOOTH point:  |AD-fwd| and |AD-bwd| both shrink ~O(eps),",
          flush=True)
    print("                      |AD-ctr| shrinks ~O(eps^2).", flush=True)
    print("  At a KINK:          AD == fwd to machine precision, while |AD-bwd|",
          flush=True)
    print("                      is eps-INDEPENDENT and |AD-ctr| = |AD-bwd|/2.",
          flush=True)
    print(flush=True)

    f = _geom_fn(pack, support, curves_jax)
    flat0 = pack.flat0
    q0 = f(flat0)
    keys = list(SMOOTH_KEYS) + list(SUSPECT_KEYS)

    n_probe = pack.n if max_dofs <= 0 else min(max_dofs, pack.n)
    worst = {"dof": -1, "key": "", "rel": 0.0}
    per_dof = []

    for j in range(n_probe):
        e_j = jnp.zeros(pack.n).at[j].set(1.0)
        _, ad = jax.jvp(f, (flat0,), (e_j,))

        print(f"  --- dof {j}: {pack.label[j]} ---", flush=True)
        print(f"      {'quantity':<10} {'eps':>8}  {'|AD-fwd|':>11} "
              f"{'|AD-bwd|':>11} {'|AD-ctr|':>11}", flush=True)
        rows = {k: [] for k in keys}
        for eps in eps_list:
            qp = f(flat0 + eps * e_j)
            qm = f(flat0 - eps * e_j)
            for k in keys:
                fwd = (qp[k] - q0[k]) / eps
                bwd = (q0[k] - qm[k]) / eps
                ctr = (qp[k] - qm[k]) / (2.0 * eps)
                rows[k].append((eps, _rel(ad[k], fwd), _rel(ad[k], bwd),
                                _rel(ad[k], ctr)))
        for k in keys:
            tag = "" if k in SMOOTH_KEYS else "  <- frame-derived"
            for i, (eps, rf, rb, rc) in enumerate(rows[k]):
                name = k if i == 0 else ""
                suffix = tag if i == 0 else ""
                print(f"      {name:<10} {eps:>8.0e}  {rf:>11.3e} {rb:>11.3e} "
                      f"{rc:>11.3e}{suffix}", flush=True)
            # eps-independence of the backward one-sided error is the tell.
            rb_first, rb_last = rows[k][0][2], rows[k][-1][2]
            plateau = rb_last > 0.1 * rb_first and rb_last > 1e-9
            per_dof.append({"dof": j, "key": k, "plateau": plateau,
                            "rel_bwd": rb_last})
            if rb_last > worst["rel"]:
                worst = {"dof": j, "key": k, "rel": rb_last}
        print(flush=True)

    kinked = sorted({r["key"] for r in per_dof if r["plateau"]})
    kinked_dofs = sorted({r["dof"] for r in per_dof if r["plateau"]})
    print("=== D2 VERDICT ===", flush=True)
    print(f"  quantities with an eps-independent one-sided error: "
          f"{kinked or 'none'}", flush=True)
    print(f"  dofs affected: {len(kinked_dofs)} / {n_probe}", flush=True)
    if worst["dof"] >= 0:
        print(f"  worst: dof {worst['dof']} ({pack.label[worst['dof']]}) "
              f"on {worst['key']}, rel={worst['rel']:.3e}", flush=True)
    if not kinked:
        print("  PASS -- geometry chain is smooth: every one-sided error scales", flush=True)
        print("  O(eps) and every centered error O(eps^2).  AD equals the true", flush=True)
        print("  derivative of the coded model.", flush=True)
    elif set(kinked) <= set(SUSPECT_KEYS):
        print("  FAIL -- kinks confined to the frame-derived quantities and absent", flush=True)
        print("  from the gamma_eval-derived ones.  The frame is not smooth in phi;", flush=True)
        print("  the defect is in the BEAM-STIFFNESS path (L_eff), not the", flush=True)
        print("  attachment-weight path.", flush=True)
    else:
        print("  FAIL -- kinks reach the gamma_eval-derived quantities too, so the", flush=True)
        print("  frame interpolant is not the only non-smooth element.", flush=True)
    return worst


# ============================================================================
# D3 -- kink scan
# ============================================================================

def phase_scan(support, pack, curves_jax, args, worst_dof):
    print(f"\n{'#' * 78}\n### D3: kink scan of L_eff across quadpoint knots"
          f"\n{'#' * 78}\n", flush=True)

    j = args.scan_dof if args.scan_dof >= 0 else worst_dof
    if j < 0 or j >= pack.n:
        print(f"  no valid scan dof (got {j}); skipping.", flush=True)
        return
    tmpls = getattr(support, "_coil_framed_templates", None)
    if tmpls is None:
        print("  meshes not bound; skipping.", flush=True)
        return

    N = int(tmpls[pack.curve_idx[j]].curve.quadpoints.shape[0])
    dphi = 1.0 / N
    f = _geom_fn(pack, support, curves_jax)
    phi0 = float(pack.flat0[j])

    # Pick the beam this dof actually moves.  Beam layout is group-dependent
    # (_beam_offsets, plus the stellsym wrap group), so select by sensitivity
    # rather than trying to reproduce the indexing.
    e_j = jnp.zeros(pack.n).at[j].set(1.0)
    _, ad = jax.jvp(f, (pack.flat0,), (e_j,))
    sens = np.abs(np.asarray(ad["L_eff"]))
    b = args.beam if args.beam >= 0 else int(np.argmax(sens))
    print(f"  beam {b} selected ({'explicit' if args.beam >= 0 else 'auto'}); "
          f"|dL_eff/dphi| = {sens[b]:.6e}  (max over beams "
          f"{sens.max():.6e})", flush=True)
    if sens[b] <= 0.0:
        print("  this dof does not move the selected beam's L_eff; "
              "pick another --beam.", flush=True)

    offs = np.linspace(-args.scan_knots * dphi, args.scan_knots * dphi,
                       args.scan_points)
    L_eff, xi_s, xi_e = [], [], []
    for d in offs:
        q = f(pack.flat0.at[j].add(float(d)))
        L_eff.append(float(q["L_eff"][b]))
        xi_s.append(float(q["xi_start"][b]))
        xi_e.append(float(q["xi_end"][b]))
    L_eff = np.asarray(L_eff)
    xi_s = np.asarray(xi_s)
    xi_e = np.asarray(xi_e)

    slope = np.gradient(L_eff, offs)
    jumps = np.abs(np.diff(slope))
    scale = max(float(np.max(np.abs(slope))), 1e-300)

    print(f"  dof {j} ({pack.label[j]}), beam {b}, phi0={phi0:.12f}", flush=True)
    print(f"  N={N}  knot spacing={dphi:.6e}  scan half-width="
          f"{args.scan_knots} knots", flush=True)
    print(f"  L_eff range: [{L_eff.min():.9e}, {L_eff.max():.9e}]", flush=True)
    print(f"  slope range: [{slope.min():.6e}, {slope.max():.6e}]", flush=True)
    print(f"  largest slope jump / max|slope|: "
          f"{float(jumps.max()) / scale:.3e}", flush=True)

    # Report the slope on each side of every knot crossed by the scan.
    print(f"\n  {'offset/dphi':>12} {'phi':>16} {'L_eff':>16} {'d L_eff/d phi':>16}",
          flush=True)
    step = max(1, args.scan_points // 21)
    for i in range(0, args.scan_points, step):
        print(f"  {offs[i] / dphi:>12.4f} {phi0 + offs[i]:>16.10f} "
              f"{L_eff[i]:>16.9e} {slope[i]:>16.6e}", flush=True)

    if args.save:
        np.savez(args.save, offs=offs, phi=phi0 + offs, L_eff=L_eff,
                 xi_start=xi_s, xi_end=xi_e, slope=slope, N=N, dof=j, beam=b)
        print(f"\n  wrote {args.save}", flush=True)

    print("\n=== D3 VERDICT ===", flush=True)
    span = float(L_eff.max() - L_eff.min())
    if span <= 1e-12 * max(abs(float(L_eff[0])), 1e-300):
        print("  L_eff is constant over this window -- the scanned dof does not "
              "move this beam.", flush=True)
        print("  Re-run with an explicit --beam, or a --scan-dof that drives it.",
              flush=True)
    elif float(jumps.max()) / scale > 1e-3:
        print("  L_eff slope has step discontinuities -- the interpolant is C0 "
              "in phi.", flush=True)
        print("  Expect the steps at integer multiples of the knot spacing "
              "above.", flush=True)
    else:
        print("  No slope steps detected in this window.", flush=True)


# ============================================================================

def main():
    args = parse_args()
    print("geom_smoothness.py -- Tier-0 (no FEM solve)", flush=True)
    print(f"  phase    = {args.phase}", flush=True)
    print(f"  n_coils  = {args.n_coils}", flush=True)
    print(f"  coupling = {args.coupling}", flush=True)
    print(f"  eps      = {args.eps}", flush=True)

    Jstress, cs = build_geometry_only(args.n_coils, args.coupling)
    fem = Jstress.fem
    support = fem.support
    curves_jax = fem.base_curves_jax
    sdofs = cs.support_dofs
    pack = PhiPack(sdofs, support)
    print(f"  phi dofs = {pack.n}", flush=True)

    worst = {"dof": -1}
    if args.phase in ("all", "knots"):
        phase_knots(support, pack)
    if args.phase in ("all", "fd"):
        worst = phase_fd(support, pack, curves_jax, args.eps, args.max_dofs)
    if args.phase in ("all", "scan"):
        phase_scan(support, pack, curves_jax, args, worst.get("dof", -1))

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
