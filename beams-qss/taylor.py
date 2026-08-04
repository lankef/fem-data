# Taylor / gradient probes for the support-only setup in ./proto.py.
#
# Baseline: centered-difference Taylor test on CoilFEMObjective.J/dJ
# (simsopt free support DOFs), same problem as proto.py.
#
# Extra probes for the residual ~2% analytic-vs-FD gap after the
# beam_geometry VJP fix:
#
#   1) --force-kt-adjoint
#        Force cuDSS general matrix type so the adjoint uses Kᵀ (solver_KT)
#        instead of reusing forward K.  If the Taylor error collapses, the
#        merged system is not exactly symmetric and adjoint_reuses_K is wrong.
#
#   2) --drop-winkler-wa
#        Zero the grounded coil Winkler term k_attachment*w_a (keep clamps
#        k_clamp*w_g and all K_cs/K_sc/K_ss beam coupling).  If the gap
#        changes sharply, the dual w_a paths (Winkler k vs beam coupling)
#        are inconsistent under AD.
#
#   3) --simsopt-free
#        Bypass simsopt Derivative: jax.grad on fem.objective w.r.t. the
#        free support flat vector, vs FD, vs Jstress.dJ().  Isolates
#        simsopt flatten/mask issues from the FEM custom_vjp.
#
#   4) --compare-wa-slice
#        One job: full model + no-w_a model.  Prints FD/analytic directional
#        derivatives and the grounded-w_a slices
#          dJh_wa ≈ FD_full - FD_no_wa
#          dJh_wa_analytic ≈ dJh_full - dJh_no_wa
#
#   5) --vjp-ablation {freeze_k|freeze_sdofs_geom}
#        Sets COIL_FEM_VJP_ABLATION before CoilFEM construction (see
#        coil-fem notes/WINKLER_WA_VJP.md).  freeze_k zeros g_k; freeze_sdofs_geom
#        stop_gradients sdofs inside constraint geom/coupling.
#
# Examples:
#   python -u ./taylor.py
#   python -u ./taylor.py --force-kt-adjoint
#   python -u ./taylor.py --drop-winkler-wa
#   python -u ./taylor.py --simsopt-free
#   python -u ./taylor.py --compare-wa-slice
#   python -u ./taylor.py --vjp-ablation freeze_k
#   python -u ./taylor.py --force-kt-adjoint --drop-winkler-wa --simsopt-free

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

# Import coil_fem before gmsh.  Loading gmsh first can pull the module
# anaconda libstdc++ ahead of the conda env's, breaking basix (CXXABI_1.3.15).
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
        help="Probe 4: full vs no-w_a FD/analytic; print grounded-w_a slices.",
    )
    p.add_argument(
        "--vjp-ablation",
        choices=("none", "freeze_k", "freeze_sdofs_geom"),
        default="none",
        help="Probe 5: set COIL_FEM_VJP_ABLATION before building CoilFEM.",
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
        "--n-coils",
        type=int,
        default=5,
        help="Number of W7-X base coils (1 fits a local GPU for smoke Taylor).",
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
        # 0 = general → adjoint_reuses_forward_K is False → build solver_KT.
        problem_options["cudss_mtype_id"] = 0

    material_options = {
        "E": 205000000000,
        "nu": 0.3,
        "density": 8000,
        "itc": 0.0,
    }
    gravity_options = {"g_vec": (0, 0, 0)}
    # One-coil local runs: keep a wrap CC beam under stellsym when possible;
    # otherwise use a foundation beam so grounded w_a stays active.
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
    """Probe 2: grounded Winkler ignores beam-attachment weight w_a."""
    support = Jstress.fem.support
    k_clamp = float(support.k_clamp)
    k_att = float(support.k_attachment)

    def stiffness_g_only(w_g, w_a):
        # Keep signature; drop k_attachment * w_a.  K_cs/K_sc/K_ss still use
        # k_attachment via coupling_values / support_values.
        return k_clamp * w_g

    support.stiffness = stiffness_g_only
    print(
        f"Probe 2: dropped winkler w_a term "
        f"(k_clamp={k_clamp:.4e}, k_attachment={k_att:.4e} still used in K_cs/K_ss).",
        flush=True,
    )


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
    """Centered-difference Taylor table for a (J, dJ) callable on free dofs."""
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
    best = None  # (abs_rel_err, eps, fd)
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
            print(
                f"    (err ratio vs previous eps: {err / prev_err:.3f}, "
                f"expect ~1e2 if O(eps^2))",
                flush=True,
            )
        prev_err = err
        last_fd = fd
        cand = (rel, eps, fd)
        if best is None or cand[0] < best[0]:
            best = cand
    if last_fd is not None and abs(last_fd) > 0:
        print(
            f"\nsummary: dJh/FD = {dJh / last_fd:.8f}  "
            f"(1.0 = perfect; baseline residual was ~0.979)",
            flush=True,
        )
    # Prefer the eps with smallest relative error for slice summaries.
    fd_best = best[2] if best is not None else last_fd
    eps_best = best[1] if best is not None else None
    if eps_best is not None:
        print(
            f"best-eps summary: eps={eps_best:.1e}  FD={fd_best:.10e}  "
            f"dJh/FD={dJh / fd_best:.8f}",
            flush=True,
        )
    return dJh, fd_best, dJ0


def make_simsopt_fun(Jstress):
    def fun(dofs):
        Jstress.x = dofs
        return Jstress.J(), Jstress.dJ()

    return fun


def make_simsopt_free_fun(Jstress, coil_support):
    """jax.grad through fem.objective on free support DOFs only (no Derivative)."""
    free_mask = np.asarray(coil_support.local_dofs_free_status, dtype=bool)
    # Concrete integer indices — boolean masks under JIT raise
    # NonConcreteBooleanIndexError for .at[].set().
    free_idx = np.flatnonzero(free_mask)
    x_full0 = np.asarray(coil_support.local_full_x, dtype=float)
    # Capture fixed curve/current values once (they are fix_all()'d).
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

    # Warm / compile grad once outside the Taylor loop.
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
    print(f"\n----- Grad compare: {name_a} vs {name_b} -----", flush=True)
    print(f"  ||a||={np.linalg.norm(a):.6e}  ||b||={np.linalg.norm(b):.6e}", flush=True)
    print(f"  ||a-b||/scale = {np.linalg.norm(diff) / scale:.6e}", flush=True)
    print(
        f"  max|a-b| = {np.max(np.abs(diff)):.6e}  "
        f"cos = {float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300):.8f}",
        flush=True,
    )


def print_wa_slice(dJh_full, fd_full, dJh_no_wa, fd_no_wa):
    """Print grounded-w_a directional-derivative slice diagnostics."""
    dJh_wa = float(fd_full) - float(fd_no_wa)
    dJh_wa_analytic = float(dJh_full) - float(dJh_no_wa)
    print(f"\n{'#' * 80}", flush=True)
    print("### Grounded-w_a slice (compare-wa-slice)", flush=True)
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
    print(
        f"  analytic/FD slice = {dJh_wa_analytic / dJh_wa:.8f}  "
        f"(|diff|/scale={abs(dJh_wa_analytic - dJh_wa) / scale:.6e})",
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


def main():
    args = parse_args()
    print("taylor.py probes:", flush=True)
    print(f"  force_kt_adjoint = {args.force_kt_adjoint}", flush=True)
    print(f"  drop_winkler_wa  = {args.drop_winkler_wa}", flush=True)
    print(f"  simsopt_free     = {args.simsopt_free}", flush=True)
    print(f"  compare_wa_slice = {args.compare_wa_slice}", flush=True)
    print(f"  vjp_ablation     = {args.vjp_ablation}", flush=True)
    print(f"  n_coils          = {args.n_coils}", flush=True)
    print(f"  eps              = {args.eps}", flush=True)
    print(f"  seed             = {args.seed}", flush=True)

    if args.vjp_ablation != "none":
        os.environ[_VJP_ABLATION_ENV] = args.vjp_ablation
        print(
            f"Set {_VJP_ABLATION_ENV}={args.vjp_ablation} "
            "(read at make_merged_solve / CoilFEM construction).",
            flush=True,
        )
    else:
        # Avoid a stale env from a previous interactive session.
        os.environ.pop(_VJP_ABLATION_ENV, None)

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

    rng = np.random.default_rng(args.seed)
    h = rng.uniform(size=dofs.shape)

    # --- Baseline / primary: simsopt J/dJ Taylor ---
    fun_simsopt = make_simsopt_fun(Jstress)
    name = "simsopt CoilFEMObjective.J/dJ"
    if args.compare_wa_slice:
        name = "FULL model (with grounded w_a Winkler) " + name
    dJh_s, fd_s, dJ_s = run_taylor(name, fun_simsopt, dofs, h, args.eps)

    # --- Probe 4: grounded-w_a slice (full vs no-w_a in one job) ---
    if args.compare_wa_slice:
        Jstress.x = dofs
        drop_winkler_wa(Jstress)
        fun_no_wa = make_simsopt_fun(Jstress)
        dJh_n, fd_n, _ = run_taylor(
            "NO grounded-w_a Winkler (k = k_clamp*w_g only) simsopt J/dJ",
            fun_no_wa,
            dofs,
            h,
            args.eps,
        )
        if fd_s is None or fd_n is None:
            raise RuntimeError("compare-wa-slice needs FD from both Taylor tables")
        print_wa_slice(dJh_s, fd_s, dJh_n, fd_n)

    # --- Probe 3: simsopt-free jax.grad path ---
    if args.simsopt_free:
        # Taylor loop leaves Jstress.x at the last perturbation; restore.
        Jstress.x = dofs
        # If compare-wa-slice already dropped w_a, rebuild so simsopt-free
        # matches the primary model the user asked for.
        if args.compare_wa_slice and not args.drop_winkler_wa:
            print(
                "Rebuilding problem for --simsopt-free on FULL model "
                "(compare-wa-slice had patched stiffness).",
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
        dJh_j, fd_j, dJ_j = run_taylor(
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
