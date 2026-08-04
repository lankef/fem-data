# Taylor / gradient probes for the support-only setup in ./proto.py.
#
# Baseline: centered-difference Taylor test on CoilFEMObjective.J/dJ
# (simsopt free support DOFs), same problem as proto.py.
#
# Extra probes for the residual ~1–2% analytic-vs-FD gap after the
# beam_geometry VJP fix:
#
#   1) --force-kt-adjoint
#   2) --drop-winkler-wa
#   3) --simsopt-free
#   4) --compare-wa-slice   (rebuilds a fresh problem for no-w_a; no mid-JIT patch)
#   5) --vjp-ablation {freeze_k|freeze_sdofs_geom|freeze_wa_in_k|freeze_wa_in_coupling}
#   6) --diagnostics        (5-coil kitchen-sink: fingerprint, full/no-wa,
#                            wa-slice, DOF-group masks, k(φ) check, simsopt-free)
#   7) --diagnostics-deep PHASE  (phased deep suite; see notes/WINKLER_WA_VJP.md)
#        groups | residual | ablation | ablation-sum | surgical | ift | scale
#
# Examples:
#   python -u ./taylor.py
#   python -u ./taylor.py --diagnostics
#   python -u ./taylor.py --diagnostics-deep groups
#   python -u ./taylor.py --diagnostics-deep residual
#   python -u ./taylor.py --diagnostics-deep ablation --vjp-ablation freeze_k
#   python -u ./taylor.py --compare-wa-slice
#   python -u ./taylor.py --drop-winkler-wa
#   sbatch jobscript_taylor.sh --diagnostics
#   sbatch jobscript_taylor.sh --diagnostics-deep residual

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

# Import coil_fem before gmsh.  Loading gmsh first can pull the module
# anaconda libstdc++ ahead of the conda env's, breaking basix (CXXABI_1.3.15).
import coil_fem
from coil_fem.coupling import drivers as coil_fem_drivers
from coil_fem.simsopt import CoilFEMObjective, CoilSupportBeams
import gmsh  # noqa: F401
from simsopt.configs import get_data
from simsopt.field import Coil
from simsopt.mhd import Vmec

jax.config.update("jax_enable_x64", True)

_VJP_ABLATION_ENV = "COIL_FEM_VJP_ABLATION"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--force-kt-adjoint",
        action="store_true",
        help="Probe 1: cudss_mtype_id=0 so adjoint uses Kᵀ, not reused K.",
    )
    p.add_argument(
        "--drop-winkler-wa",
        action="store_true",
        help="Probe 2: coil Winkler uses only k_clamp*w_g (drop k_attachment*w_a).",
    )
    p.add_argument(
        "--simsopt-free",
        action="store_true",
        help="Probe 3: also run jax.grad(fem.objective) Taylor vs simsopt dJ.",
    )
    p.add_argument(
        "--compare-wa-slice",
        action="store_true",
        help="Probe 4: full vs no-w_a (separate rebuilt problems); print wa slices.",
    )
    p.add_argument(
        "--vjp-ablation",
        choices=(
            "none",
            "freeze_k",
            "freeze_sdofs_geom",
            "freeze_wa_in_k",
            "freeze_wa_in_coupling",
        ),
        default="none",
        help="Probe 5: set COIL_FEM_VJP_ABLATION before building CoilFEM.",
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Probe 6: full 5-coil diagnostic suite (see module docstring).",
    )
    p.add_argument(
        "--diagnostics-deep",
        choices=(
            "groups",
            "residual",
            "ablation",
            "ablation-sum",
            "surgical",
            "ift",
            "scale",
        ),
        default=None,
        help="Probe 7: phased deep diagnostics (one phase per job).",
    )
    p.add_argument(
        "--eps",
        type=float,
        nargs="+",
        default=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        help="Centered-difference step sizes.",
    )
    p.add_argument("--seed", type=int, default=1, help="RNG seed for direction h.")
    p.add_argument(
        "--normalize-h",
        action="store_true",
        help="Scale h to unit Euclidean norm (default: off, match historical jobs).",
    )
    p.add_argument(
        "--n-coils",
        type=int,
        default=5,
        help="Number of W7-X base coils (default 5).",
    )
    p.add_argument(
        "--ablation-dir",
        type=str,
        default="logs",
        help="Directory for deep-ablation JSON artifacts (ablation / ablation-sum).",
    )
    return p.parse_args()


def build_problem(force_kt_adjoint: bool, n_coils: int = 5):
    """Construct CoilSupportBeams + CoilFEMObjective (proto.py setup)."""
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

    mesh_options = {
        "shape": "rect",
        "w1": 0.2,
        "w2": 0.2,
        "frame": "rmf",
        "aspect_ratio": 1.0,
        "mesh_type": "TET10",
    }
    problem_options = {
        "solver": "cudss",
        "adjoint_solver": "cudss",
    }
    if force_kt_adjoint:
        problem_options["cudss_mtype_id"] = 0

    material_options = {
        "E": 205000000000,
        "nu": 0.3,
        "density": 8000,
        "itc": 0.0,
    }
    gravity_options = {"g_vec": (0, 0, 0)}
    if n_coils == 1:
        beam_options = {
            "n_beam_cc": 1,
            "n_beam_cf": 1,
            "E": material_options["E"],
            "nu": material_options["nu"],
            "cross_section_type": "solid_circle",
            "attachment_type": "direct",
        }
    else:
        beam_options = {
            "n_beam_cc": 4,
            "n_beam_cf": 0,
            "E": material_options["E"],
            "nu": material_options["nu"],
            "cross_section_type": "solid_circle",
            "attachment_type": "direct",
        }
    fixed_clamp_options = {
        "enabled": True,
        "r_clamp": 1.73 * mesh_options["w1"] / 2,
        "n_clamp": 2,
        "E_coil": material_options["E"],
    }
    physics_options = {"type": "elastic"}

    base_coils = [Coil(c, I) for c, I in zip(base_curves, base_currents)]
    coil_support = CoilSupportBeams(
        base_coils=base_coils,
        nfp=eq.boundary.nfp,
        stellsym=eq.boundary.stellsym,
        beam_options=beam_options,
        r_beam=0.06,
        fixed_clamp_options=fixed_clamp_options,
        fixed_dof_names=(
            "thetas_orientation_cc",
            "r_beam",
        ),
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
        coupling="monolithic",
    )

    for c in base_curves:
        c.fix_all()
    for cur in base_currents:
        cur.fix_all()

    return Jstress, coil_support


def drop_winkler_wa(Jstress):
    """Grounded Winkler ignores beam-attachment weight w_a (apply before JIT)."""
    support = Jstress.fem.support
    k_clamp = float(support.k_clamp)
    k_att = float(support.k_attachment)

    def stiffness_g_only(w_g, w_a):
        return k_clamp * w_g

    support.stiffness = stiffness_g_only
    print(
        f"Dropped winkler w_a term "
        f"(k_clamp={k_clamp:.4e}, k_attachment={k_att:.4e} still used in K_cs/K_ss).",
        flush=True,
    )


def print_coil_fem_fingerprint():
    """Print imported coil-fem paths and whether drivers.py contains k_live."""
    drivers_path = coil_fem_drivers.__file__
    try:
        src = open(drivers_path, encoding="utf-8", errors="ignore").read()
        has_k_live = "k_live" in src
    except OSError as exc:
        has_k_live = f"<read error: {exc}>"
    print("----- coil-fem fingerprint -----", flush=True)
    print(f"  coil_fem.__file__              : {coil_fem.__file__}", flush=True)
    print(f"  coil_fem.coupling.drivers file : {drivers_path}", flush=True)
    print(f"  k_live_in_drivers              : {has_k_live}", flush=True)
    print(
        f"  COIL_FEM_VJP_ABLATION          : "
        f"{os.environ.get(_VJP_ABLATION_ENV, '')!r}",
        flush=True,
    )
    return {
        "coil_fem": coil_fem.__file__,
        "drivers": drivers_path,
        "k_live_in_drivers": has_k_live,
        "vjp_ablation_env": os.environ.get(_VJP_ABLATION_ENV, ""),
    }


def print_adjoint_diagnostics(Jstress, force_kt_adjoint: bool):
    static = Jstress.fem.monolithic_static
    print("----- Probe 1 diagnostics (adjoint K vs Kᵀ) -----", flush=True)
    print(f"  force_kt_adjoint flag : {force_kt_adjoint}", flush=True)
    if static is None:
        print("  monolithic_static     : None", flush=True)
        return
    print(f"  adjoint_reuses_K      : {static.adjoint_reuses_K}", flush=True)
    print(f"  solver_KT is None     : {static.solver_KT is None}", flush=True)
    print(
        f"  coo_to_csr_T is None  : {static.coo_to_csr_T is None}",
        flush=True,
    )
    if force_kt_adjoint and static.adjoint_reuses_K:
        print(
            "  WARNING: expected adjoint_reuses_K=False with cudss_mtype_id=0.",
            flush=True,
        )
    if (not force_kt_adjoint) and (not static.adjoint_reuses_K):
        print(
            "  NOTE: baseline is already using Kᵀ (not reusing forward K).",
            flush=True,
        )


def run_taylor(name, J_and_dJ, dofs, h, eps_list):
    """Centered-difference Taylor table for a (J, dJ) callable on free dofs.

    Returns
    -------
    dJh, fd_plateau, fd_best, dJ0, meta
        ``fd_plateau`` is the last (smallest) eps FD — use for systematic-bias
        ratios.  ``fd_best`` is the eps with smallest |err|/|dJh|.
    """
    print(f"\n{'#' * 80}\n### Taylor test: {name}\n{'#' * 80}\n", flush=True)
    J0, dJ0 = J_and_dJ(dofs)
    dJ0 = np.asarray(dJ0, dtype=float).reshape(-1)
    h = np.asarray(h, dtype=float).reshape(-1)
    dJh = float(np.dot(dJ0, h))
    print(f"J0  = {J0}", flush=True)
    print(f"dJh = {dJh}", flush=True)
    print(f"|dJ| = {np.linalg.norm(dJ0)}", flush=True)
    print(flush=True)
    print(
        f"{'eps':>10} {'centered diff':>18} {'err':>18} {'err/eps^2':>18} "
        f"{'|err|/|dJh|':>14}",
        flush=True,
    )
    prev_err = None
    last_fd = None
    last_eps = None
    best = None  # (abs_rel_err, eps, fd)
    rows = []
    for eps in eps_list:
        J1, _ = J_and_dJ(dofs + eps * h)
        J2, _ = J_and_dJ(dofs - eps * h)
        fd = (J1 - J2) / (2 * eps)
        err = fd - dJh
        rel = abs(err) / max(abs(dJh), 1.0)
        print(
            f"{eps:10.1e} {fd:18.10e} {err:18.10e} {err / eps**2:18.10e} {rel:14.6e}",
            flush=True,
        )
        if prev_err is not None:
            ratio = err / prev_err if prev_err != 0 else float("nan")
            print(
                f"    (err ratio vs previous eps: {ratio:.3f}, "
                f"expect ~1e2 if O(eps^2); ~1 if systematic bias)",
                flush=True,
            )
        rows.append({"eps": eps, "fd": fd, "err": err, "rel": rel})
        prev_err = err
        last_fd = fd
        last_eps = eps
        cand = (rel, eps, fd)
        if best is None or cand[0] < best[0]:
            best = cand
    fd_best = best[2] if best is not None else last_fd
    eps_best = best[1] if best is not None else None
    if last_fd is not None and abs(last_fd) > 0:
        print(
            f"\nsummary (plateau/last eps={last_eps:.1e}): dJh/FD = {dJh / last_fd:.8f}  "
            f"(1.0 = perfect; W7-X residual was ~0.988)",
            flush=True,
        )
    if eps_best is not None and abs(fd_best) > 0:
        print(
            f"best-eps summary: eps={eps_best:.1e}  FD={fd_best:.10e}  "
            f"dJh/FD={dJh / fd_best:.8f}",
            flush=True,
        )
    meta = {
        "J0": float(J0),
        "dJh": dJh,
        "fd_plateau": last_fd,
        "eps_plateau": last_eps,
        "fd_best": fd_best,
        "eps_best": eps_best,
        "rows": rows,
    }
    return dJh, last_fd, fd_best, dJ0, meta


def make_simsopt_fun(Jstress):
    def fun(dofs):
        Jstress.x = dofs
        return Jstress.J(), Jstress.dJ()

    return fun


def make_simsopt_free_fun(Jstress, coil_support):
    """jax.grad through fem.objective on free support DOFs only (no Derivative)."""
    free_mask = np.asarray(coil_support.local_dofs_free_status, dtype=bool)
    free_idx = np.flatnonzero(free_mask)
    x_full0 = np.asarray(coil_support.local_full_x, dtype=float)
    cdofs0 = [jnp.asarray(c.get_dofs()) for c in coil_support.base_curves]
    idofs0 = jnp.asarray(
        [c.get_value() for c in coil_support.base_currents], dtype=jnp.float64
    )
    unravel = coil_support._unravel
    fem = Jstress.fem
    weight = float(Jstress._metric_weights[0])
    metric = Jstress._metrics[0]

    def J_free(x_free):
        x_full = jnp.asarray(x_full0).at[free_idx].set(x_free)
        sdofs = unravel(x_full)
        out = fem.objective(cdofs0, idofs0, sdofs, metrics=(metric,))
        return weight * out[metric]

    grad_J = jax.jit(jax.grad(J_free))
    J_jit = jax.jit(J_free)

    def fun(dofs):
        x = jnp.asarray(dofs, dtype=jnp.float64)
        return float(J_jit(x)), np.asarray(grad_J(x), dtype=float)

    return fun, free_mask


def compare_grads(name_a, dJa, name_b, dJb):
    a = np.asarray(dJa, dtype=float).reshape(-1)
    b = np.asarray(dJb, dtype=float).reshape(-1)
    diff = a - b
    scale = max(np.linalg.norm(a), np.linalg.norm(b), 1.0)
    rel = float(np.linalg.norm(diff) / scale)
    print(f"\n----- Grad compare: {name_a} vs {name_b} -----", flush=True)
    print(f"  ||a||={np.linalg.norm(a):.6e}  ||b||={np.linalg.norm(b):.6e}", flush=True)
    print(f"  ||a-b||/scale = {rel:.6e}", flush=True)
    print(
        f"  max|a-b| = {np.max(np.abs(diff)):.6e}  "
        f"cos = {float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300):.8f}",
        flush=True,
    )
    return rel


def print_wa_slice(dJh_full, fd_full, dJh_no_wa, fd_no_wa, *, label="plateau"):
    """Print grounded-w_a directional-derivative slice diagnostics."""
    dJh_wa = float(fd_full) - float(fd_no_wa)
    dJh_wa_analytic = float(dJh_full) - float(dJh_no_wa)
    gap_full = abs(float(dJh_full) - float(fd_full))
    print(f"\n{'#' * 80}", flush=True)
    print(f"### Grounded-w_a slice ({label} FD)", flush=True)
    print(f"{'#' * 80}\n", flush=True)
    print(f"  dJh_full          = {dJh_full:.16e}", flush=True)
    print(f"  FD_full           = {fd_full:.16e}", flush=True)
    print(f"  dJh_full/FD_full  = {dJh_full / fd_full:.8f}", flush=True)
    print(f"  dJh_no_wa         = {dJh_no_wa:.16e}", flush=True)
    print(f"  FD_no_wa          = {fd_no_wa:.16e}", flush=True)
    print(f"  dJh_no_wa/FD_no_wa= {dJh_no_wa / fd_no_wa:.8f}", flush=True)
    print(f"  dJh_wa            = FD_full - FD_no_wa     = {dJh_wa:.16e}", flush=True)
    print(
        f"  dJh_wa_analytic   = dJh_full - dJh_no_wa   = {dJh_wa_analytic:.16e}",
        flush=True,
    )
    scale = max(abs(dJh_wa), abs(dJh_wa_analytic), 1.0)
    slice_ratio = dJh_wa_analytic / dJh_wa if dJh_wa != 0 else float("nan")
    print(
        f"  analytic/FD slice = {slice_ratio:.8f}  "
        f"(|diff|/scale={abs(dJh_wa_analytic - dJh_wa) / scale:.6e})",
        flush=True,
    )
    if abs(dJh_wa) > 0:
        print(
            f"  |full gap| / |FD wa slice| = {gap_full / abs(dJh_wa):.8f}",
            flush=True,
        )
    if abs(dJh_wa_analytic) < 0.5 * abs(dJh_wa):
        verdict = "UNDER-differentiated grounded-w_a path (most likely)"
    elif abs(dJh_wa_analytic) > 1.5 * abs(dJh_wa):
        verdict = "OVER-differentiated / double-counted grounded-w_a path"
    elif dJh_wa * dJh_wa_analytic < 0:
        verdict = "WRONG-SIGN grounded-w_a path"
    else:
        verdict = "slice magnitudes comparable (look at ratio carefully)"
    print(f"  interpretation    : {verdict}", flush=True)
    return {
        "dJh_wa": dJh_wa,
        "dJh_wa_analytic": dJh_wa_analytic,
        "slice_ratio": slice_ratio,
        "gap_over_wa_fd": gap_full / abs(dJh_wa) if dJh_wa != 0 else float("nan"),
    }


def free_group_mask(coil_support, key: str) -> np.ndarray:
    """Boolean mask over free DOFs corresponding to support_dofs[key]."""
    free = np.asarray(coil_support.local_dofs_free_status, dtype=bool)
    x0 = np.asarray(coil_support.local_full_x, dtype=float)
    sdofs0 = coil_support._unravel(jnp.asarray(x0))
    if key not in sdofs0:
        return np.zeros(int(free.sum()), dtype=bool)
    ind = jax.tree_util.tree_map(jnp.zeros_like, sdofs0)
    ind = dict(ind)
    ind[key] = jax.tree_util.tree_map(jnp.ones_like, sdofs0[key])
    flat = np.asarray(ravel_pytree(ind)[0], dtype=float)
    if flat.shape != free.shape:
        raise RuntimeError(
            f"indicator flat shape {flat.shape} != local_full_x mask {free.shape}"
        )
    return flat[free] > 0.5


def make_direction_h(dofs, seed, normalize: bool):
    rng = np.random.default_rng(seed)
    h = rng.uniform(size=dofs.shape)
    if normalize:
        nrm = np.linalg.norm(h)
        if nrm > 0:
            h = h / nrm
    return h


def run_k_phi_grad_check(Jstress, coil_support, dofs, h, eps=1e-6):
    """FD vs jax.grad of sum(k) along free-support direction (no merged_solve)."""
    print(f"\n{'#' * 80}\n### k(φ) grad check (outside merged_solve)\n{'#' * 80}\n", flush=True)
    free_mask = np.asarray(coil_support.local_dofs_free_status, dtype=bool)
    free_idx = np.flatnonzero(free_mask)
    x_full0 = np.asarray(coil_support.local_full_x, dtype=float)
    unravel = coil_support._unravel
    fem = Jstress.fem
    cdofs0 = [jnp.asarray(c.get_dofs()) for c in coil_support.base_curves]
    curves_live = fem.curves_from_dofs(cdofs0)
    pts0 = [
        fem.meshes[i].mesh_points_from_dofs(cdofs0[i])
        for i in range(len(cdofs0))
    ]

    def sum_k(x_free):
        x_full = jnp.asarray(x_full0).at[free_idx].set(x_free)
        sdofs = unravel(x_full)
        geom = fem.support.beam_geometry(curves_live, sdofs)
        total = 0.0
        for i in range(len(cdofs0)):
            w_g, w_a = fem._support_weights(
                i, pts0[i], curves_live, sdofs, geom=geom,
            )
            total = total + jnp.sum(fem.support.stiffness(w_g, w_a))
        return total

    x0 = jnp.asarray(dofs, dtype=jnp.float64)
    h_j = jnp.asarray(h, dtype=jnp.float64)
    g = jax.grad(sum_k)(x0)
    d_an = float(jnp.dot(g, h_j))
    d_fd = float(
        (sum_k(x0 + eps * h_j) - sum_k(x0 - eps * h_j)) / (2.0 * eps)
    )
    ratio = d_an / d_fd if d_fd != 0 else float("nan")
    print(f"  d(sum k)/dh analytic = {d_an:.16e}", flush=True)
    print(f"  d(sum k)/dh FD       = {d_fd:.16e}", flush=True)
    print(f"  analytic/FD          = {ratio:.8f}", flush=True)
    return ratio


def run_dof_group_probes(Jstress, coil_support, dofs, h, eps_list, dJ_full):
    """Taylor along h masked to individual support DOF groups (full multi-ε)."""
    print(f"\n{'#' * 80}\n### DOF-group directional probes\n{'#' * 80}\n", flush=True)
    fun = make_simsopt_fun(Jstress)
    keys = ("phis_start_cc", "phis_end_cc", "phis")
    masks = {k: free_group_mask(coil_support, k) for k in keys}
    covered = np.zeros(dofs.shape, dtype=bool)
    for m in masks.values():
        covered |= m
    masks["other"] = ~covered

    dJ_full = np.asarray(dJ_full, dtype=float).reshape(-1)
    h = np.asarray(h, dtype=float).reshape(-1)
    dJh_full_dir = float(np.dot(dJ_full, h))
    eps_lo = float(min(eps_list))
    group_ratios = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            print(f"  {name:16s}  n_free=0  (skip)", flush=True)
            group_ratios[name] = float("nan")
            continue
        h_g = h * mask.astype(float)
        Jstress.x = dofs
        dJh_g, fd_g, _, _, meta_g = run_taylor(
            f"DOF-group {name}",
            fun,
            dofs,
            h_g,
            eps_list,
        )
        ratio = dJh_g / fd_g if fd_g else float("nan")
        frac = dJh_g / dJh_full_dir if dJh_full_dir != 0 else float("nan")
        print(
            f"  {name:16s}  n_free={n:3d}  dJh={dJh_g:.6e}  "
            f"FD(plateau/eps={eps_lo:.0e})={fd_g:.6e}  dJh/FD={ratio:.8f}  "
            f"dJh/dJh_full={frac:.6f}",
            flush=True,
        )
        group_ratios[name] = ratio
        Jstress.x = dofs
    return group_ratios


def _ablation_json_path(ablation_dir: str, ablation: str, n_coils: int, seed: int) -> Path:
    return Path(ablation_dir) / f"deep_ablation_{ablation}_n{n_coils}_s{seed}.json"


def _sol_flat_from_run(fem, out):
    """Concatenate coil displacements + support DOFs into merged sol_flat."""
    static = fem.monolithic_static
    parts = []
    for i, u in enumerate(out["displacements"]):
        parts.append(jnp.asarray(u).reshape(-1))
    u_s = out["u_s"]
    if u_s is None:
        raise RuntimeError("expected coupled solve with u_s")
    parts.append(jnp.asarray(u_s).reshape(-1))
    sol = jnp.concatenate(parts)
    if int(sol.shape[0]) != int(static.n_total_dofs):
        raise RuntimeError(
            f"sol_flat length {sol.shape[0]} != n_total_dofs {static.n_total_dofs}"
        )
    return sol


def _pack_free_helpers(coil_support, dofs):
    free_mask = np.asarray(coil_support.local_dofs_free_status, dtype=bool)
    free_idx = np.flatnonzero(free_mask)
    x_full0 = np.asarray(coil_support.local_full_x, dtype=float)
    unravel = coil_support._unravel
    x0 = jnp.asarray(dofs, dtype=jnp.float64)

    def sdofs_from_free(x_free):
        x_full = jnp.asarray(x_full0).at[free_idx].set(x_free)
        return unravel(x_full)

    return free_idx, x_full0, sdofs_from_free, x0


def run_deep_groups(args):
    """Phase 0: full-model Taylor + multi-ε DOF-group localization."""
    fp = print_coil_fem_fingerprint()
    J, cs = build_problem(args.force_kt_adjoint, n_coils=args.n_coils)
    print_adjoint_diagnostics(J, args.force_kt_adjoint)
    dofs = np.asarray(J.x, dtype=float)
    h = make_direction_h(dofs, args.seed, args.normalize_h)
    fun = make_simsopt_fun(J)
    dJh, fd, _, dJ, _ = run_taylor("DEEP groups: full model", fun, dofs, h, args.eps)
    J.x = dofs
    group_ratios = run_dof_group_probes(J, cs, dofs, h, args.eps, dJ)
    ratio = dJh / fd if fd else float("nan")
    print("\n=== DEEP DIAG PHASE groups SUMMARY ===", flush=True)
    print(f"fingerprint: drivers={fp['drivers']}", flush=True)
    print(f"k_live_in_drivers: {fp['k_live_in_drivers']}", flush=True)
    print(f"n_coils: {args.n_coils}", flush=True)
    print(f"baseline_dJh_over_FD: {ratio:.8f}", flush=True)
    print(
        "group_ratios: "
        + " ".join(f"{k}={v:.8f}" for k, v in group_ratios.items()),
        flush=True,
    )


def run_deep_residual(args):
    """Phase 1: frozen-u* residual FD vs VJP (coil / coupling / full)."""
    from coil_fem.solvers.cudss import assemble_csr_values

    fp = print_coil_fem_fingerprint()
    J, cs = build_problem(args.force_kt_adjoint, n_coils=args.n_coils)
    dofs = np.asarray(J.x, dtype=float)
    h = make_direction_h(dofs, args.seed, args.normalize_h)
    J.x = dofs
    fem = J.fem
    support = fem.support
    static = fem.monolithic_static
    if static is None or static.merged_solve is None:
        raise RuntimeError("deep residual requires monolithic cuDSS merged_solve")

    out = J.run()
    sol_flat = _sol_flat_from_run(fem, out)
    cdofs0 = [jnp.asarray(c.get_dofs()) for c in cs.base_curves]
    curves_live0 = fem.curves_from_dofs(cdofs0)
    n_base = len(cdofs0)
    pts0 = [fem.meshes[i].mesh_points_from_dofs(cdofs0[i]) for i in range(n_base)]
    bf0 = [jnp.asarray(out["f_vol"][i]) for i in range(n_base)]
    fe_geom0 = []
    for i in range(n_base):
        fe_geom0.append(fem._jit_fe_geom_fns[i](pts0[i]))

    free_idx, x_full0, sdofs_from_free, x0 = _pack_free_helpers(cs, dofs)
    h_j = jnp.asarray(h, dtype=jnp.float64)

    coil_dof_offsets = static.coil_dof_offsets
    n_dofs_per_coil = static.n_dofs_per_coil
    support_dof_offset = static.support_dof_offset
    n_s = static.n_s
    has_cs = static.has_cs
    has_sc = static.has_sc
    I_cs = jnp.asarray(static.I_cs_pat) if has_cs else None
    J_cs = jnp.asarray(static.J_cs_pat) if has_cs else None
    I_sc = jnp.asarray(static.I_sc_pat) if has_sc else None
    J_sc = jnp.asarray(static.J_sc_pat) if has_sc else None
    pipelines = fem.pipelines
    surf_interp = [
        (
            pipelines[i].problem._sel_face_sv,
            pipelines[i].problem._surf_face_to_surf_node,
            int(pipelines[i].problem._surf_unique_global_nodes.shape[0]),
        )
        for i in range(n_base)
    ]

    def _k_list(sdofs, geom):
        ks = []
        for i in range(n_base):
            w_g, w_a = fem._support_weights(
                i, pts0[i], curves_live0, sdofs, geom=geom,
            )
            ks.append(support.stiffness(w_g, w_a))
        return ks

    def R_coil_live_k(x_free):
        sdofs = sdofs_from_free(x_free)
        geom = support.beam_geometry(curves_live0, sdofs)
        k_list = _k_list(sdofs, geom)
        residuals = []
        for i, pipeline in enumerate(pipelines):
            p_par = {
                "points": pts0[i],
                "body_force": bf0[i],
                "support_k": k_list[i],
                "_fe_geom": fe_geom0[i],
            }
            u_c_i = sol_flat[
                coil_dof_offsets[i]:coil_dof_offsets[i] + n_dofs_per_coil[i]
            ].reshape(pipeline.problem.fes[0].num_total_nodes, 3)
            pipeline.problem.set_params(p_par)
            res_i = pipeline.problem.compute_residual_vars(
                [u_c_i],
                pipeline.problem.internal_vars,
                pipeline.problem.internal_vars_surfaces,
            )
            residuals.append(ravel_pytree(res_i)[0])
        return jnp.concatenate(residuals)

    def R_coil_frozen_k(x_free, k_frozen):
        residuals = []
        for i, pipeline in enumerate(pipelines):
            p_par = {
                "points": pts0[i],
                "body_force": bf0[i],
                "support_k": k_frozen[i],
                "_fe_geom": fe_geom0[i],
            }
            u_c_i = sol_flat[
                coil_dof_offsets[i]:coil_dof_offsets[i] + n_dofs_per_coil[i]
            ].reshape(pipeline.problem.fes[0].num_total_nodes, 3)
            pipeline.problem.set_params(p_par)
            res_i = pipeline.problem.compute_residual_vars(
                [u_c_i],
                pipeline.problem.internal_vars,
                pipeline.problem.internal_vars_surfaces,
            )
            residuals.append(ravel_pytree(res_i)[0])
        return jnp.concatenate(residuals)

    def R_coupling(x_free):
        sdofs = sdofs_from_free(x_free)
        geom = support.beam_geometry(curves_live0, sdofs)
        geom_kw = {"geom": geom}
        s_quad = [pipelines[i].surface_quad_points(pts0[i]) for i in range(n_base)]
        jxw = [pipelines[i].problem.surface_jxw(pts0[i]) for i in range(n_base)]
        Iss, Jss = support.support_pattern()
        Vss = support.support_values(
            curves_live0, sdofs, s_quad, **geom_kw, jxw_by_coil=jxw,
        )
        u_s = sol_flat[support_dof_offset:]
        r_s = jnp.zeros(n_s, dtype=Vss.dtype).at[Iss].add(Vss * u_s[Jss])
        n_c = int(support_dof_offset)
        r_full = jnp.concatenate([jnp.zeros(n_c, dtype=r_s.dtype), r_s])
        V_cs, V_sc = support.coupling_values(
            curves_live0, sdofs, s_quad,
            surf_interp_by_coil=surf_interp,
            jxw_by_coil=jxw,
            **geom_kw,
        )
        if has_cs:
            r_full = r_full.at[I_cs].add(V_cs * sol_flat[J_cs])
        if has_sc:
            r_full = r_full.at[I_sc].add(V_sc * sol_flat[J_sc])
        return r_full

    def R_full(x_free):
        r_c = R_coil_live_k(x_free)
        r_k = R_coupling(x_free)
        # R_coupling already padded to full length; coil block of r_k is zero.
        return r_k.at[: r_c.shape[0]].add(r_c)

    sdofs0 = sdofs_from_free(x0)
    geom0 = support.beam_geometry(curves_live0, sdofs0)
    k0 = _k_list(sdofs0, geom0)

    key = jax.random.PRNGKey(7 + int(args.seed))
    v_coil = jax.random.normal(key, R_coil_live_k(x0).shape)
    v_coup = jax.random.normal(jax.random.fold_in(key, 1), R_coupling(x0).shape)
    v_full = jax.random.normal(jax.random.fold_in(key, 2), R_full(x0).shape)

    # Also build adjoint λ from ∂J/∂u at frozen φ (support-only: ∂J/∂φ|_u ≈ 0).
    metric = J._metrics[0]
    if metric != "l2_von_mises":
        raise NotImplementedError(
            f"deep residual λ cotangent only for l2_von_mises, got {metric}"
        )
    weight = float(J._metric_weights[0])
    from coil_fem.metrics import l2_von_mises

    def J_of_sol(sol):
        totals = 0.0
        for i, pipeline in enumerate(pipelines):
            n_nodes = pipeline.problem.fes[0].num_total_nodes
            u = sol[
                coil_dof_offsets[i]:coil_dof_offsets[i] + n_dofs_per_coil[i]
            ].reshape(n_nodes, 3)
            p_par = {
                "points": pts0[i],
                "body_force": bf0[i],
                "support_k": k0[i],
                "_fe_geom": fe_geom0[i],
            }
            pipeline.problem.set_params(p_par)
            sg, jxw, _, _ = fe_geom0[i]
            totals = totals + l2_von_mises(
                pipeline.problem, [u], fem._lam, fem._mu,
                shape_grads=sg, JxW=jxw,
            )
        return weight * totals

    g_u = jax.grad(J_of_sol)(sol_flat)

    # Build V_merged by calling the same blocks as forward assemble.
    s_quad0 = [pipelines[i].surface_quad_points(pts0[i]) for i in range(n_base)]
    jxw0 = [pipelines[i].problem.surface_jxw(pts0[i]) for i in range(n_base)]
    V_blocks = []
    for i, pipeline in enumerate(pipelines):
        p_par = {
            "points": pts0[i],
            "body_force": bf0[i],
            "support_k": k0[i],
            "_fe_geom": fe_geom0[i],
        }
        _, _, Vi, _, _ = pipeline.assemble_coo(p_par)
        V_blocks.append(Vi)
    Iss, Jss = support.support_pattern()
    Vss = support.support_values(
        curves_live0, sdofs0, s_quad0, geom=geom0, jxw_by_coil=jxw0,
    )
    V_blocks.append(Vss)
    V_cs, V_sc = support.coupling_values(
        curves_live0, sdofs0, s_quad0,
        surf_interp_by_coil=surf_interp,
        jxw_by_coil=jxw0,
        geom=geom0,
    )
    if has_cs:
        V_blocks.append(V_cs)
    if has_sc:
        V_blocks.append(V_sc)
    V_merged = jnp.concatenate([jnp.asarray(v) for v in V_blocks])
    if static.adjoint_reuses_K:
        csr_values = assemble_csr_values(V_merged, static.coo_to_csr, static.nnz_csr)
        lambda_flat, _ = static.solver_K(g_u, csr_values)
    else:
        csr_values_T = assemble_csr_values(
            V_merged, static.coo_to_csr_T, static.nnz_csr_T,
        )
        lambda_flat, _ = static.solver_KT(g_u, csr_values_T)

    print(f"\n{'#' * 80}\n### DEEP residual FD vs VJP at frozen u*\n{'#' * 80}\n", flush=True)
    print(f"  ||sol||={float(jnp.linalg.norm(sol_flat)):.6e}", flush=True)
    print(f"  ||g_u||={float(jnp.linalg.norm(g_u)):.6e}", flush=True)
    print(f"  ||λ||={float(jnp.linalg.norm(lambda_flat)):.6e}", flush=True)
    r0 = R_full(x0)
    print(f"  ||R(u*;φ0)||={float(jnp.linalg.norm(r0)):.6e}", flush=True)

    eps_list = list(args.eps)
    # Prefer smaller steps for residual directional probes.
    eps_use = [e for e in eps_list if e <= 1e-4] or eps_list
    eps_fd = float(min(eps_use))

    def _dir_ratio(name, sfn):
        # Directional derivative along h: d/dα s(x0 + α h)|_0
        def s_alpha(alpha):
            return sfn(x0 + alpha * h_j)

        d_an = float(jax.grad(s_alpha)(0.0))
        d_fd = float(
            (s_alpha(eps_fd) - s_alpha(-eps_fd)) / (2.0 * eps_fd)
        )
        ratio = d_an / d_fd if d_fd != 0 else float("nan")
        print(
            f"  {name:28s}  d_an={d_an:.10e}  FD({eps_fd:.0e})={d_fd:.10e}  "
            f"an/FD={ratio:.8f}",
            flush=True,
        )
        return {"d_an": d_an, "d_fd": d_fd, "ratio": ratio}

    rows = {}
    rows["R_coil_live_k_rand"] = _dir_ratio(
        "R_coil live k (rand v)",
        lambda x: jnp.dot(v_coil, R_coil_live_k(x)),
    )
    rows["R_coil_frozen_k_rand"] = _dir_ratio(
        "R_coil frozen k (rand v)",
        lambda x: jnp.dot(v_coil, R_coil_frozen_k(x, k0)),
    )
    rows["R_coupling_rand"] = _dir_ratio(
        "R_coupling (rand v)",
        lambda x: jnp.dot(v_coup, R_coupling(x)),
    )
    rows["R_full_rand"] = _dir_ratio(
        "R_full (rand v)",
        lambda x: jnp.dot(v_full, R_full(x)),
    )
    # Actual adjoint cotangent on full residual.
    rows["R_full_lambda"] = _dir_ratio(
        "R_full (λ cotangent)",
        lambda x: jnp.dot(lambda_flat, R_full(x)),
    )
    # Coil block of λ only.
    n_c = int(support_dof_offset)
    lam_c = lambda_flat[:n_c]
    rows["R_coil_live_lambda"] = _dir_ratio(
        "R_coil live k (λ_c)",
        lambda x: jnp.dot(lam_c, R_coil_live_k(x)),
    )
    rows["R_coupling_lambda"] = _dir_ratio(
        "R_coupling (λ)",
        lambda x: jnp.dot(lambda_flat, R_coupling(x)),
    )

    print("\n=== DEEP DIAG PHASE residual SUMMARY ===", flush=True)
    print(f"fingerprint: drivers={fp['drivers']}", flush=True)
    print(f"k_live_in_drivers: {fp['k_live_in_drivers']}", flush=True)
    print(f"n_coils: {args.n_coils}", flush=True)
    print(f"eps_fd: {eps_fd:.1e}", flush=True)
    for k, v in rows.items():
        print(f"  {k}: an/FD={v['ratio']:.8f}", flush=True)
    # Decision hints
    r_live = rows["R_coil_live_k_rand"]["ratio"]
    r_coup = rows["R_coupling_rand"]["ratio"]
    if abs(r_live - 1.0) > 5e-3 and np.isfinite(r_live):
        print(
            "  hint: R_coil live k disagrees → Winkler residual wiring at W7-X scale",
            flush=True,
        )
    elif abs(r_coup - 1.0) > 5e-3 and np.isfinite(r_coup):
        print(
            "  hint: R_coupling disagrees → coupling residual VJP at W7-X scale",
            flush=True,
        )
    else:
        print(
            "  hint: residual blocks healthy → look at IFT composition (phases ablation/ift)",
            flush=True,
        )


def run_deep_ablation(args):
    """Phase 2: single ablation Taylor + JSON artifact for additive sum."""
    fp = print_coil_fem_fingerprint()
    ablation = args.vjp_ablation
    J, cs = build_problem(args.force_kt_adjoint, n_coils=args.n_coils)
    dofs = np.asarray(J.x, dtype=float)
    h = make_direction_h(dofs, args.seed, args.normalize_h)
    fun = make_simsopt_fun(J)
    dJh, fd, _, dJ, meta = run_taylor(
        f"DEEP ablation={ablation}", fun, dofs, h, args.eps,
    )
    ratio = dJh / fd if fd else float("nan")
    Path(args.ablation_dir).mkdir(parents=True, exist_ok=True)
    out_path = _ablation_json_path(
        args.ablation_dir, ablation, args.n_coils, args.seed,
    )
    payload = {
        "ablation": ablation,
        "n_coils": args.n_coils,
        "seed": args.seed,
        "normalize_h": bool(args.normalize_h),
        "dJh": float(dJh),
        "fd_plateau": float(fd) if fd is not None else None,
        "dJh_over_FD": float(ratio) if np.isfinite(ratio) else None,
        "J0": float(meta["J0"]),
        "eps_plateau": float(meta["eps_plateau"]) if meta["eps_plateau"] else None,
        "k_live_in_drivers": bool(fp["k_live_in_drivers"]),
        "drivers": fp["drivers"],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote ablation artifact: {out_path}", flush=True)
    print("\n=== DEEP DIAG PHASE ablation SUMMARY ===", flush=True)
    print(f"ablation: {ablation}", flush=True)
    print(f"dJh: {dJh:.16e}", flush=True)
    print(f"FD_plateau: {fd:.16e}" if fd else "FD_plateau: None", flush=True)
    print(f"dJh_over_FD: {ratio:.8f}", flush=True)
    print(
        "Next: run freeze_k and freeze_sdofs_geom (same seed/n-coils), then "
        "--diagnostics-deep ablation-sum",
        flush=True,
    )


def run_deep_ablation_sum(args):
    """Phase 2b: combine none / freeze_k / freeze_sdofs_geom JSON artifacts."""
    print_coil_fem_fingerprint()
    names = ("none", "freeze_k", "freeze_sdofs_geom")
    data = {}
    for name in names:
        path = _ablation_json_path(
            args.ablation_dir, name, args.n_coils, args.seed,
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"missing ablation artifact {path}; "
                f"run --diagnostics-deep ablation --vjp-ablation {name} first"
            )
        data[name] = json.loads(path.read_text(encoding="utf-8"))
        print(f"  loaded {path}", flush=True)

    dA = data["none"]["dJh"]
    dB = data["freeze_k"]["dJh"]
    dC = data["freeze_sdofs_geom"]["dJh"]
    fdA = data["none"]["fd_plateau"]
    sum_bc = dB + dC
    print(f"\n{'#' * 80}\n### DEEP ablation additive reconstruction\n{'#' * 80}\n", flush=True)
    print(f"  dJh_A (none)              = {dA:.16e}", flush=True)
    print(f"  dJh_B (freeze_k)          = {dB:.16e}", flush=True)
    print(f"  dJh_C (freeze_sdofs_geom) = {dC:.16e}", flush=True)
    print(f"  FD_A                      = {fdA:.16e}", flush=True)
    print(f"  dJh_B + dJh_C             = {sum_bc:.16e}", flush=True)
    print(
        f"  (B+C)/A                   = {sum_bc / dA:.8f}" if dA else "  (B+C)/A = nan",
        flush=True,
    )
    print(
        f"  (B+C)/FD_A                = {sum_bc / fdA:.8f}" if fdA else "  (B+C)/FD_A = nan",
        flush=True,
    )
    print(
        f"  A/FD_A                    = {dA / fdA:.8f}" if fdA else "  A/FD_A = nan",
        flush=True,
    )
    den = abs(dB) + abs(dC)
    share_b = abs(dB) / den if den > 0 else float("nan")
    share_c = abs(dC) / den if den > 0 else float("nan")
    print(f"  |B|/(|B|+|C|)             = {share_b:.6f}", flush=True)
    print(f"  |C|/(|B|+|C|)             = {share_c:.6f}", flush=True)
    print("\n=== DEEP DIAG PHASE ablation-sum SUMMARY ===", flush=True)
    if abs(sum_bc - dA) / max(abs(dA), 1.0) < 1e-3:
        print("  partition: B+C ≈ A (ablations partition cleanly)", flush=True)
    else:
        print("  partition: B+C ≠ A (shared paths / interpret carefully)", flush=True)
    if fdA and abs(dA / fdA - 1.0) > 5e-3:
        print("  gap vs FD remains on full reverse (A); under-diff vs forward FD", flush=True)


def run_deep_surgical(args):
    """Phase 3: surgical w_a ablation Taylor (freeze_wa_in_k / freeze_wa_in_coupling)."""
    if args.vjp_ablation not in ("freeze_wa_in_k", "freeze_wa_in_coupling"):
        raise ValueError(
            "surgical phase expects --vjp-ablation freeze_wa_in_k "
            "or freeze_wa_in_coupling"
        )
    fp = print_coil_fem_fingerprint()
    ablation = args.vjp_ablation
    J, cs = build_problem(args.force_kt_adjoint, n_coils=args.n_coils)
    dofs = np.asarray(J.x, dtype=float)
    h = make_direction_h(dofs, args.seed, args.normalize_h)
    fun = make_simsopt_fun(J)
    dJh, fd, _, _, meta = run_taylor(
        f"DEEP surgical ablation={ablation}", fun, dofs, h, args.eps,
    )
    ratio = dJh / fd if fd else float("nan")
    Path(args.ablation_dir).mkdir(parents=True, exist_ok=True)
    out_path = _ablation_json_path(
        args.ablation_dir, ablation, args.n_coils, args.seed,
    )
    payload = {
        "ablation": ablation,
        "n_coils": args.n_coils,
        "seed": args.seed,
        "dJh": float(dJh),
        "fd_plateau": float(fd) if fd is not None else None,
        "dJh_over_FD": float(ratio) if np.isfinite(ratio) else None,
        "J0": float(meta["J0"]),
        "drivers": fp["drivers"],
        "k_live_in_drivers": bool(fp["k_live_in_drivers"]),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote surgical artifact: {out_path}", flush=True)
    print("\n=== DEEP DIAG PHASE surgical SUMMARY ===", flush=True)
    print(f"ablation: {ablation}", flush=True)
    print(f"dJh_over_FD: {ratio:.8f}", flush=True)
    print(f"J0: {meta['J0']:.16e}", flush=True)


def run_deep_ift(args):
    """Phase 4: residual norms, adjoint quality, manual IFT vs custom_vjp / FD."""
    from coil_fem.metrics import l2_von_mises
    from coil_fem.solvers.cudss import assemble_csr_values

    fp = print_coil_fem_fingerprint()
    J, cs = build_problem(args.force_kt_adjoint, n_coils=args.n_coils)
    dofs = np.asarray(J.x, dtype=float)
    h = make_direction_h(dofs, args.seed, args.normalize_h)
    fun = make_simsopt_fun(J)
    dJh_vjp, fd, _, dJ, meta = run_taylor(
        "DEEP ift: custom_vjp simsopt J/dJ", fun, dofs, h, args.eps,
    )

    J.x = dofs
    fem = J.fem
    support = fem.support
    static = fem.monolithic_static
    out = J.run()
    sol_flat = _sol_flat_from_run(fem, out)
    cdofs0 = [jnp.asarray(c.get_dofs()) for c in cs.base_curves]
    curves_live0 = fem.curves_from_dofs(cdofs0)
    n_base = len(cdofs0)
    pts0 = [fem.meshes[i].mesh_points_from_dofs(cdofs0[i]) for i in range(n_base)]
    bf0 = [jnp.asarray(out["f_vol"][i]) for i in range(n_base)]
    fe_geom0 = [fem._jit_fe_geom_fns[i](pts0[i]) for i in range(n_base)]
    free_idx, x_full0, sdofs_from_free, x0 = _pack_free_helpers(cs, dofs)
    h_j = jnp.asarray(h, dtype=jnp.float64)

    coil_dof_offsets = static.coil_dof_offsets
    n_dofs_per_coil = static.n_dofs_per_coil
    support_dof_offset = static.support_dof_offset
    n_s = static.n_s
    has_cs, has_sc = static.has_cs, static.has_sc
    I_cs = jnp.asarray(static.I_cs_pat) if has_cs else None
    J_cs = jnp.asarray(static.J_cs_pat) if has_cs else None
    I_sc = jnp.asarray(static.I_sc_pat) if has_sc else None
    J_sc = jnp.asarray(static.J_sc_pat) if has_sc else None
    pipelines = fem.pipelines
    surf_interp = [
        (
            pipelines[i].problem._sel_face_sv,
            pipelines[i].problem._surf_face_to_surf_node,
            int(pipelines[i].problem._surf_unique_global_nodes.shape[0]),
        )
        for i in range(n_base)
    ]

    def _k_list(sdofs, geom):
        return [
            support.stiffness(*fem._support_weights(
                i, pts0[i], curves_live0, sdofs, geom=geom,
            ))
            for i in range(n_base)
        ]

    def R_full(x_free):
        sdofs = sdofs_from_free(x_free)
        geom = support.beam_geometry(curves_live0, sdofs)
        k_list = _k_list(sdofs, geom)
        residuals = []
        for i, pipeline in enumerate(pipelines):
            p_par = {
                "points": pts0[i],
                "body_force": bf0[i],
                "support_k": k_list[i],
                "_fe_geom": fe_geom0[i],
            }
            u_c_i = sol_flat[
                coil_dof_offsets[i]:coil_dof_offsets[i] + n_dofs_per_coil[i]
            ].reshape(pipeline.problem.fes[0].num_total_nodes, 3)
            pipeline.problem.set_params(p_par)
            res_i = pipeline.problem.compute_residual_vars(
                [u_c_i],
                pipeline.problem.internal_vars,
                pipeline.problem.internal_vars_surfaces,
            )
            residuals.append(ravel_pytree(res_i)[0])
        s_quad = [pipelines[i].surface_quad_points(pts0[i]) for i in range(n_base)]
        jxw = [pipelines[i].problem.surface_jxw(pts0[i]) for i in range(n_base)]
        Iss, Jss = support.support_pattern()
        Vss = support.support_values(
            curves_live0, sdofs, s_quad, geom=geom, jxw_by_coil=jxw,
        )
        u_s = sol_flat[support_dof_offset:]
        r_s = jnp.zeros(n_s, dtype=Vss.dtype).at[Iss].add(Vss * u_s[Jss])
        r_full = jnp.concatenate(residuals + [r_s])
        V_cs, V_sc = support.coupling_values(
            curves_live0, sdofs, s_quad,
            surf_interp_by_coil=surf_interp,
            jxw_by_coil=jxw,
            geom=geom,
        )
        if has_cs:
            r_full = r_full.at[I_cs].add(V_cs * sol_flat[J_cs])
        if has_sc:
            r_full = r_full.at[I_sc].add(V_sc * sol_flat[J_sc])
        return r_full

    sdofs0 = sdofs_from_free(x0)
    geom0 = support.beam_geometry(curves_live0, sdofs0)
    k0 = _k_list(sdofs0, geom0)
    weight = float(J._metric_weights[0])
    if J._metrics[0] != "l2_von_mises":
        raise NotImplementedError(
            f"deep ift only for l2_von_mises, got {J._metrics[0]}"
        )

    def J_of_sol(sol):
        totals = 0.0
        for i, pipeline in enumerate(pipelines):
            n_nodes = pipeline.problem.fes[0].num_total_nodes
            u = sol[
                coil_dof_offsets[i]:coil_dof_offsets[i] + n_dofs_per_coil[i]
            ].reshape(n_nodes, 3)
            p_par = {
                "points": pts0[i],
                "body_force": bf0[i],
                "support_k": k0[i],
                "_fe_geom": fe_geom0[i],
            }
            pipeline.problem.set_params(p_par)
            sg, jxw, _, _ = fe_geom0[i]
            totals = totals + l2_von_mises(
                pipeline.problem, [u], fem._lam, fem._mu,
                shape_grads=sg, JxW=jxw,
            )
        return weight * totals

    g_u = jax.grad(J_of_sol)(sol_flat)
    s_quad0 = [pipelines[i].surface_quad_points(pts0[i]) for i in range(n_base)]
    jxw0 = [pipelines[i].problem.surface_jxw(pts0[i]) for i in range(n_base)]
    V_blocks = []
    f_blocks = []
    for i, pipeline in enumerate(pipelines):
        p_par = {
            "points": pts0[i],
            "body_force": bf0[i],
            "support_k": k0[i],
            "_fe_geom": fe_geom0[i],
        }
        _, _, Vi, _, fi = pipeline.assemble_coo(p_par)
        V_blocks.append(Vi)
        f_blocks.append(fi)
    Iss, Jss = support.support_pattern()
    Vss = support.support_values(
        curves_live0, sdofs0, s_quad0, geom=geom0, jxw_by_coil=jxw0,
    )
    V_blocks.append(Vss)
    f_blocks.append(jnp.zeros(n_s, dtype=Vss.dtype))
    V_cs, V_sc = support.coupling_values(
        curves_live0, sdofs0, s_quad0,
        surf_interp_by_coil=surf_interp,
        jxw_by_coil=jxw0,
        geom=geom0,
    )
    if has_cs:
        V_blocks.append(V_cs)
    if has_sc:
        V_blocks.append(V_sc)
    V_merged = jnp.concatenate([jnp.asarray(v) for v in V_blocks])
    f_merged = jnp.concatenate(f_blocks)
    csr_values = assemble_csr_values(V_merged, static.coo_to_csr, static.nnz_csr)
    # Forward residual check: K u - f
    # Reconstruct Ku via COO scatter (pattern from static).
    # Use residual R(u*) instead (includes same physics).
    r0 = R_full(x0)
    nrm_r = float(jnp.linalg.norm(r0))
    nrm_f = float(jnp.linalg.norm(f_merged))

    if static.adjoint_reuses_K:
        lambda_flat, _ = static.solver_K(g_u, csr_values)
    else:
        csr_values_T = assemble_csr_values(
            V_merged, static.coo_to_csr_T, static.nnz_csr_T,
        )
        lambda_flat, _ = static.solver_KT(g_u, csr_values_T)

    # Manual IFT: for support-only free DOFs, ∂J/∂φ|_u ≈ 0, so
    # dJ/dh ≈ λ · dR/dh at frozen u*.
    def s_lam(alpha):
        return jnp.dot(lambda_flat, R_full(x0 + alpha * h_j))

    dJh_manual = float(jax.grad(s_lam)(0.0))
    eps_fd = float(min(args.eps))
    fd_manual = float((s_lam(eps_fd) - s_lam(-eps_fd)) / (2.0 * eps_fd))

    print(f"\n{'#' * 80}\n### DEEP IFT identity checks\n{'#' * 80}\n", flush=True)
    print(f"  ||R(u*;φ0)||           = {nrm_r:.6e}", flush=True)
    print(f"  ||f||                  = {nrm_f:.6e}", flush=True)
    print(f"  ||R||/||f||            = {nrm_r / max(nrm_f, 1.0):.6e}", flush=True)
    print(f"  ||g_u||                = {float(jnp.linalg.norm(g_u)):.6e}", flush=True)
    print(f"  ||λ||                  = {float(jnp.linalg.norm(lambda_flat)):.6e}", flush=True)
    print(f"  adjoint_reuses_K       = {static.adjoint_reuses_K}", flush=True)
    print(f"  dJh_custom_vjp         = {dJh_vjp:.16e}", flush=True)
    print(f"  FD_J (plateau)         = {fd:.16e}", flush=True)
    print(f"  dJh_manual_IFT (λ·∂R)  = {dJh_manual:.16e}", flush=True)
    print(f"  FD(λ·R)                = {fd_manual:.16e}", flush=True)
    print(
        f"  manual/custom_vjp      = {dJh_manual / dJh_vjp:.8f}"
        if dJh_vjp else "  manual/custom_vjp = nan",
        flush=True,
    )
    print(
        f"  manual/FD_J            = {dJh_manual / fd:.8f}" if fd else "  manual/FD_J = nan",
        flush=True,
    )
    print(
        f"  custom_vjp/FD_J        = {dJh_vjp / fd:.8f}" if fd else "  custom_vjp/FD_J = nan",
        flush=True,
    )
    print(
        f"  (λ·∂R VJP)/(λ·R FD)    = {dJh_manual / fd_manual:.8f}"
        if fd_manual else "  (λ·∂R VJP)/(λ·R FD) = nan",
        flush=True,
    )

    print("\n=== DEEP DIAG PHASE ift SUMMARY ===", flush=True)
    print(f"fingerprint: drivers={fp['drivers']}", flush=True)
    print(f"k_live_in_drivers: {fp['k_live_in_drivers']}", flush=True)
    print(f"n_coils: {args.n_coils}", flush=True)
    print(f"||R||/||f||: {nrm_r / max(nrm_f, 1.0):.6e}", flush=True)
    if fd:
        print(f"custom_vjp_over_FD: {dJh_vjp / fd:.8f}", flush=True)
        print(f"manual_IFT_over_FD: {dJh_manual / fd:.8f}", flush=True)
    if dJh_vjp:
        print(f"manual_over_custom_vjp: {dJh_manual / dJh_vjp:.8f}", flush=True)
    if fd and abs(dJh_manual / fd - 1.0) < 5e-3 and abs(dJh_vjp / fd - 1.0) > 5e-3:
        print(
            "  hint: manual IFT matches FD but custom_vjp does not → "
            "g_k/g_sdofs stitching in _bwd/_solve_all",
            flush=True,
        )
    elif fd and abs(dJh_manual / fd - 1.0) > 5e-3:
        print(
            "  hint: manual IFT also short vs FD_J → direct φ dependence or "
            "residual definition mismatch under load",
            flush=True,
        )


def run_deep_scale(args):
    """Phase 5: full Taylor at the requested --n-coils (compare 1 vs 5 across jobs)."""
    fp = print_coil_fem_fingerprint()
    J, cs = build_problem(args.force_kt_adjoint, n_coils=args.n_coils)
    dofs = np.asarray(J.x, dtype=float)
    h = make_direction_h(dofs, args.seed, args.normalize_h)
    fun = make_simsopt_fun(J)
    dJh, fd, _, _, meta = run_taylor(
        f"DEEP scale n_coils={args.n_coils}", fun, dofs, h, args.eps,
    )
    ratio = dJh / fd if fd else float("nan")
    print("\n=== DEEP DIAG PHASE scale SUMMARY ===", flush=True)
    print(f"fingerprint: drivers={fp['drivers']}", flush=True)
    print(f"k_live_in_drivers: {fp['k_live_in_drivers']}", flush=True)
    print(f"n_coils: {args.n_coils}", flush=True)
    print(f"J0: {meta['J0']:.16e}", flush=True)
    print(f"dJh_over_FD: {ratio:.8f}", flush=True)
    print(
        "Compare this ratio across --n-coils 1 and --n-coils 5 (same seed).",
        flush=True,
    )


def run_diagnostics_deep(args):
    """Dispatch --diagnostics-deep PHASE."""
    phase = args.diagnostics_deep
    print(f"\n===== diagnostics-deep phase={phase} =====\n", flush=True)
    if phase == "groups":
        run_deep_groups(args)
    elif phase == "residual":
        run_deep_residual(args)
    elif phase == "ablation":
        run_deep_ablation(args)
    elif phase == "ablation-sum":
        run_deep_ablation_sum(args)
    elif phase == "surgical":
        run_deep_surgical(args)
    elif phase == "ift":
        run_deep_ift(args)
    elif phase == "scale":
        run_deep_scale(args)
    else:
        raise ValueError(f"unknown diagnostics-deep phase: {phase}")


def run_diagnostics(args):
    """Full diagnostic suite (default 5 coils)."""
    fp = print_coil_fem_fingerprint()

    # --- Full model ---
    J_full, cs_full = build_problem(args.force_kt_adjoint, n_coils=args.n_coils)
    print_adjoint_diagnostics(J_full, args.force_kt_adjoint)
    dofs = np.asarray(J_full.x, dtype=float)
    h = make_direction_h(dofs, args.seed, args.normalize_h)
    print("# mesh node for all coils:", J_full.n_nodes, flush=True)
    print("# mesh cell for all coils:", J_full.n_cells, flush=True)
    print("# free dofs:", dofs.shape, flush=True)
    print(f"|h| = {np.linalg.norm(h):.16e}", flush=True)
    print("dof names  :", J_full.dof_names, flush=True)

    fun_full = make_simsopt_fun(J_full)
    dJh_full, fd_full, fd_best_full, dJ_full, meta_full = run_taylor(
        "DIAG full model simsopt J/dJ",
        fun_full,
        dofs,
        h,
        args.eps,
    )

    # --- No-w_a model (rebuild; never mid-JIT patch) ---
    J_no, cs_no = build_problem(args.force_kt_adjoint, n_coils=args.n_coils)
    drop_winkler_wa(J_no)
    # Align free-x with full model (same optimizable layout).
    J_no.x = dofs.copy()
    fun_no = make_simsopt_fun(J_no)
    J0_no_check, _ = fun_no(dofs)
    if abs(meta_full["J0"] - J0_no_check) < 1e-6 * max(abs(meta_full["J0"]), 1.0):
        raise RuntimeError(
            f"no-w_a rebuild did not change J0 "
            f"(full={meta_full['J0']}, no_wa={J0_no_check}); "
            "drop_winkler_wa may have been applied too late / wrong object."
        )
    print(
        f"Rebuild check: J0_full={meta_full['J0']:.16e}  "
        f"J0_no_wa={J0_no_check:.16e}  "
        f"rel_diff={abs(meta_full['J0'] - J0_no_check) / max(abs(meta_full['J0']), 1.0):.6e}",
        flush=True,
    )

    dJh_no, fd_no, _, dJ_no, meta_no = run_taylor(
        "DIAG no grounded-w_a simsopt J/dJ",
        fun_no,
        dofs,
        h,
        args.eps,
    )
    slice_info = print_wa_slice(
        dJh_full, fd_full, dJh_no, fd_no, label="plateau/last-eps",
    )

    # --- DOF groups on full model ---
    J_full.x = dofs
    group_ratios = run_dof_group_probes(
        J_full, cs_full, dofs, h, args.eps, dJ_full,
    )

    # --- k(φ) outside custom_vjp ---
    J_full.x = dofs
    k_ratio = run_k_phi_grad_check(J_full, cs_full, dofs, h)

    # --- simsopt vs jax.grad ---
    J_full.x = dofs
    fun_jax, free_mask = make_simsopt_free_fun(J_full, cs_full)
    assert free_mask.sum() == dofs.size
    dJh_j, fd_j, _, dJ_j, _ = run_taylor(
        "DIAG simsopt-free jax.grad(fem.objective)",
        fun_jax,
        dofs,
        h,
        args.eps,
    )
    simsopt_vs_jax = compare_grads(
        "simsopt dJ", dJ_full, "jax.grad fem.objective", dJ_j,
    )

    baseline_ratio = dJh_full / fd_full if fd_full else float("nan")
    no_wa_ratio = dJh_no / fd_no if fd_no else float("nan")

    print("\n=== DIAGNOSTICS SUMMARY ===", flush=True)
    print(f"fingerprint: drivers={fp['drivers']}", flush=True)
    print(f"k_live_in_drivers: {fp['k_live_in_drivers']}", flush=True)
    print(f"vjp_ablation_env: {fp['vjp_ablation_env']!r}", flush=True)
    print(f"n_coils: {args.n_coils}", flush=True)
    print(f"baseline_dJh_over_FD: {baseline_ratio:.8f}", flush=True)
    print(f"no_wa_dJh_over_FD: {no_wa_ratio:.8f}", flush=True)
    print(
        f"wa_slice_analytic_over_FD: {slice_info['slice_ratio']:.8f}",
        flush=True,
    )
    print(
        f"full_gap_over_wa_fd_slice: {slice_info['gap_over_wa_fd']:.8f}",
        flush=True,
    )
    print(f"k_grad_over_FD: {k_ratio:.8f}", flush=True)
    print(
        "group_ratios: "
        + " ".join(f"{k}={v:.8f}" for k, v in group_ratios.items()),
        flush=True,
    )
    print(f"simsopt_vs_jax_rel: {simsopt_vs_jax:.6e}", flush=True)
    print(
        f"simsopt_free_dJh_over_FD: {dJh_j / fd_j if fd_j else float('nan'):.8f}",
        flush=True,
    )


def main():
    args = parse_args()
    print("taylor.py probes:", flush=True)
    print(f"  force_kt_adjoint = {args.force_kt_adjoint}", flush=True)
    print(f"  drop_winkler_wa  = {args.drop_winkler_wa}", flush=True)
    print(f"  simsopt_free     = {args.simsopt_free}", flush=True)
    print(f"  compare_wa_slice = {args.compare_wa_slice}", flush=True)
    print(f"  vjp_ablation     = {args.vjp_ablation}", flush=True)
    print(f"  diagnostics      = {args.diagnostics}", flush=True)
    print(f"  diagnostics_deep = {args.diagnostics_deep}", flush=True)
    print(f"  n_coils          = {args.n_coils}", flush=True)
    print(f"  normalize_h      = {args.normalize_h}", flush=True)
    print(f"  eps              = {args.eps}", flush=True)
    print(f"  seed             = {args.seed}", flush=True)

    # ablation-sum only reads JSON; do not force an ablation env.
    if args.diagnostics_deep == "ablation-sum":
        os.environ.pop(_VJP_ABLATION_ENV, None)
    elif args.vjp_ablation != "none":
        os.environ[_VJP_ABLATION_ENV] = args.vjp_ablation
        print(
            f"Set {_VJP_ABLATION_ENV}={args.vjp_ablation} "
            "(read at make_merged_solve / CoilFEM construction).",
            flush=True,
        )
    else:
        os.environ.pop(_VJP_ABLATION_ENV, None)

    if args.diagnostics_deep is not None:
        if args.diagnostics:
            print(
                "NOTE: --diagnostics-deep takes precedence over --diagnostics.",
                flush=True,
            )
        run_diagnostics_deep(args)
        print("\nDone.", flush=True)
        return

    if args.diagnostics:
        run_diagnostics(args)
        print("\nDone.", flush=True)
        return

    print_coil_fem_fingerprint()

    Jstress, coil_support = build_problem(
        args.force_kt_adjoint, n_coils=args.n_coils,
    )
    print("# mesh node for all coils:", Jstress.n_nodes, flush=True)
    print("# mesh cell for all coils:", Jstress.n_cells, flush=True)

    print_adjoint_diagnostics(Jstress, args.force_kt_adjoint)

    if args.drop_winkler_wa and not args.compare_wa_slice:
        drop_winkler_wa(Jstress)

    dofs = np.asarray(Jstress.x, dtype=float)
    print("# free dofs:", dofs.shape, flush=True)
    print("dof names  :", Jstress.dof_names, flush=True)

    h = make_direction_h(dofs, args.seed, args.normalize_h)
    print(f"|h| = {np.linalg.norm(h):.16e}", flush=True)

    # --- Baseline / primary: simsopt J/dJ Taylor ---
    fun_simsopt = make_simsopt_fun(Jstress)
    name = "simsopt CoilFEMObjective.J/dJ"
    if args.compare_wa_slice:
        name = "FULL model (with grounded w_a Winkler) " + name
    dJh_s, fd_s, fd_best_s, dJ_s, meta_s = run_taylor(
        name, fun_simsopt, dofs, h, args.eps,
    )

    # --- Probe 4: grounded-w_a slice (rebuild no-w_a; never patch after JIT) ---
    if args.compare_wa_slice:
        J_no, _ = build_problem(args.force_kt_adjoint, n_coils=args.n_coils)
        drop_winkler_wa(J_no)
        J_no.x = dofs.copy()
        fun_no_wa = make_simsopt_fun(J_no)
        J0_no, _ = fun_no_wa(dofs)
        if abs(meta_s["J0"] - J0_no) < 1e-6 * max(abs(meta_s["J0"]), 1.0):
            raise RuntimeError(
                f"compare-wa-slice: no-w_a J0≈full J0 ({J0_no}); rebuild failed"
            )
        dJh_n, fd_n, _, _, _ = run_taylor(
            "NO grounded-w_a Winkler (k = k_clamp*w_g only) simsopt J/dJ",
            fun_no_wa,
            dofs,
            h,
            args.eps,
        )
        print_wa_slice(dJh_s, fd_s, dJh_n, fd_n, label="plateau/last-eps")

    # --- Probe 3: simsopt-free jax.grad path ---
    if args.simsopt_free:
        Jstress.x = dofs
        if args.compare_wa_slice and not args.drop_winkler_wa:
            print(
                "Rebuilding problem for --simsopt-free on FULL model.",
                flush=True,
            )
            Jstress, coil_support = build_problem(
                args.force_kt_adjoint, n_coils=args.n_coils,
            )
            Jstress.x = dofs
        fun_jax, free_mask = make_simsopt_free_fun(Jstress, coil_support)
        assert free_mask.sum() == dofs.size, (
            f"free mask size {free_mask.sum()} != Jstress.x size {dofs.size}"
        )
        dJh_j, fd_j, _, dJ_j, _ = run_taylor(
            "simsopt-free jax.grad(fem.objective)",
            fun_jax,
            dofs,
            h,
            args.eps,
        )
        compare_grads("simsopt dJ", dJ_s, "jax.grad fem.objective", dJ_j)
        if fd_s is not None and fd_j is not None:
            print(
                f"\nFD compare: simsopt_FD={fd_s:.10e}  jax_FD={fd_j:.10e}  "
                f"rel_diff={abs(fd_s - fd_j) / max(abs(fd_s), abs(fd_j), 1.0):.6e}",
                flush=True,
            )
            print(
                "Interpretation:\n"
                "  - simsopt dJ ≈ jax.grad but both ≠ FD  → FEM VJP (issues 1–2)\n"
                "  - simsopt dJ ≠ jax.grad               → simsopt flatten/Derivative\n"
                "  - jax.grad ≈ FD, simsopt ≠ FD         → simsopt path only",
                flush=True,
            )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
