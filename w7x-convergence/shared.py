"""Shared utilities and configuration for W7-X coil FEM benchmark scripts."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: F401 – re-exported for caller use
print(f"JAX devices: {jax.devices()}")
# Enable 64-bit precision for better accuracy
jax.config.update("jax_enable_x64", True)

# ── Coilforce / simsopt imports ───────────────────────────────────────────────
from coilforce.curve_jax import CurveXYZFourierJAX
from coilforce.framed_curve_jax import make_centroid_frame, make_rmf_frame  # noqa: F401
from coilforce.meshing import rectangle_sweep, disk_sweep  # noqa: F401
from coilforce.coil_fem import CoilFEM

# ── Physics / solver options ──────────────────────────────────────────────────
# Rectangular cross-section (half-widths in metres).
mesh_options = dict(
    shape        = 'rect',
    w1           = 0.2,   # 0.20 m half-width
    w2           = 0.2,   # 0.20 m half-width
    frame        = 'rmf',
    aspect_ratio = 1.0,   # aim for cubic elements
    mesh_type    = 'TET10',
)

# Winkler BC and linear-solver settings.
problem_options = dict(
    winkler_k      = 1e10,      # Winkler spring stiffness [N/m³]
    # solver         = 'jax',     # JAX solver. (GPU). Default solver is UMFPack (CPU)
    # adjoint_solver = 'jax',
)

# ─────────────────────────────────────────────────────────────────────────────
# Material, thermal-contraction and gravity parameters are centralised in
# ``fem-data/properties.json`` (repo root) and loaded here so every script uses
# the same values + literature references.  W7-X coil casings / support
# structure are AISI 316LN austenitic stainless steel; values (E, nu, density,
# 293→4 K integral thermal contraction) are from Foussat et al. 2013 (see
# properties.json).
# To disable thermal or gravity, edit properties.json (set gravity.enabled=false
# or drop the itc key); no code changes needed.
# ─────────────────────────────────────────────────────────────────────────────
_PROPERTIES_PATH = Path(__file__).resolve().parent.parent / 'properties.json'
with open(_PROPERTIES_PATH) as _f:
    _PROPERTIES = json.load(_f)

_mat = _PROPERTIES['material']
_grav = _PROPERTIES.get('gravity', {})

# Elastic + thermal material options forwarded to CoilFEM.
material_options = dict(
    E       = float(_mat['E_Pa']),
    nu      = float(_mat['nu']),
    density = float(_mat['density_kg_m3']),
    itc     = float(_mat['itc']),   # integral thermal contraction ΔL/L; eps_th = -itc·I
)

# Gravity body-force options (None disables the gravity load).
if _grav.get('enabled', False):
    gravity_options = dict(
        density = float(_mat['density_kg_m3']),
        g_vec   = tuple(float(c) for c in _grav['g_vec_m_s2']),
    )
else:
    gravity_options = None

# ── Support-function geometry ─────────────────────────────────────────────────
# Clamp radius = 2 × coil half-width; sigmoid sharpness tuned to clamp_radius.
clamp_radius = 0.3 # 2 * max(mesh_options['w1'], mesh_options['w2'])
sigmoid_beta  = 20.0 / clamp_radius

support_dofs = None   # static support (no optimisable parameters)


def support_fn(
    surface_points: jax.Array,   # (n_surface_nodes, 3)  traced
    coil: 'CurveXYZFourierJAX',  # current coil geometry
    dofs: dict | None,           # optimisable support params (or None)
) -> jax.Array:                  # (n_surface_nodes,) weights in [0, 1]
    """Soft-sphere support at top and bottom of the coil centreline."""
    gamma  = coil.gamma()                              # (n_quad, 3)
    top    = gamma[jnp.argmax(gamma[:, 2])]            # (3,) highest point
    bottom = gamma[jnp.argmin(gamma[:, 2])]            # (3,) lowest point

    # Safe norm: jnp.linalg.norm gradient is NaN at zero distance;
    # adding eps inside sqrt keeps the backward pass finite.
    d_top    = jnp.sqrt(jnp.sum((surface_points - top)**2,    axis=-1) + 1e-30)
    d_bottom = jnp.sqrt(jnp.sum((surface_points - bottom)**2, axis=-1) + 1e-30)

    # sigmoid(beta*(R-d)): ~1 inside sphere of radius clamp_radius, ~0 outside
    w_top    = jax.nn.sigmoid(sigmoid_beta * (clamp_radius - d_top))
    w_bottom = jax.nn.sigmoid(sigmoid_beta * (clamp_radius - d_bottom))
    return jnp.maximum(w_top, w_bottom)   # union of the two spheres


# ── Benchmark quadpoints sweep ────────────────────────────────────────────────
N_QUADPOINTS_LIST = [100, 125, 150, 175, 200, 225, 250, 275]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_w7x_data():
    """Load W7-X coil geometry from simsopt; returns (curves, currents, axis, nfp, bs)."""
    from simsopt.configs import get_data
    return get_data('w7x')


# ── SLURM array task helpers ──────────────────────────────────────────────────
def parse_task_args() -> tuple[int, int]:
    """Parse --task_id and --num_tasks from the command line (SLURM array arguments)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task_id',   type=int, default=0,
                        help='SLURM array task index (0-based)')
    parser.add_argument('--num_tasks', type=int, default=1,
                        help='Total number of array tasks (SLURM_ARRAY_TASK_COUNT)')
    args, _ = parser.parse_known_args()
    return args.task_id, args.num_tasks


def get_task_items(task_id: int) -> list[int]:
    """Return the subset of N_QUADPOINTS_LIST assigned to *this* task via stride slicing."""
    return N_QUADPOINTS_LIST[task_id]


# ── Core FEM runner (one quadpoints resolution) ───────────────────────────────
def run_one_n_quadpoints(
    n_quadpoints_i: int,
    base_curve_objs: list,   # simsopt curve objects (have .get_dofs() / .order)
    base_current_objs: list, # simsopt current objects (have .current attribute)
    nfp: int,
    stellsym: bool,
    save_dir: str,
    profile=False,
    solver='cudss',
    run=True
) -> tuple[dict | None, 'CoilFEM | None', float, float]:
    """
    Run the FEM solve for *one* quadpoints resolution.

    Returns
    -------
    result  : dict of JAX arrays (or None on OOM / skip)
    t_jit   : compile + first-run time in seconds (NaN on skip)
    t_run   : warm run time in seconds (NaN on skip)
    """
    dir_i       = f'n_{n_quadpoints_i}'
    out_dir     = Path(save_dir) / dir_i
    timing_file = out_dir / 'timings.npz'
    out_dir.mkdir(parents=True, exist_ok=True)

    quadpoints_new = jnp.linspace(0, 1, n_quadpoints_i, endpoint=False)

    fem = CoilFEM(
        base_curves_jax      = [
            CurveXYZFourierJAX(quadpoints=quadpoints_new,
                               dofs=c.get_dofs(), order=c.order)
            for c in base_curve_objs
        ],
        base_currents_jax    = [c.current for c in base_current_objs],
        nfp              = nfp,
        stellsym         = stellsym,
        mesh_options     = mesh_options,
        material_options = material_options,
        gravity_options  = gravity_options,
        base_support_fns = support_fn,
        base_support_dofs = support_dofs,
        problem_options  = problem_options|{'solver':solver, 'adjoint_solver':solver},
    )
    fem.save_support_vtu(str(out_dir), prefix="coil_")
    result = None
    t_jit  = float('nan')
    t_run  = float('nan')
    jax.clear_caches()

    if not run:
        return None, fem, None, None
        
    t0     = time.perf_counter()
    result = fem.run()
    jax.block_until_ready(result)
    t_jit  = time.perf_counter() - t0

    t0     = time.perf_counter()
    if profile:
        with jax.profiler.trace("/scratch/lf2869/project/fem/w7x-convergence/jax_trace_" + solver, create_perfetto_link=False):
            result = fem.run()
            jax.block_until_ready(result)
    else:
        result = fem.run()
        jax.block_until_ready(result)
    t_run  = time.perf_counter() - t0

    fem.save_run_vtu(str(out_dir), prefix="coil_")
    np.savez(
        timing_file,
        t_jit=t_jit,
        t_run=t_run,
        **{k: np.array(v) for k, v in result.items()},
    )

    return result, fem, t_jit, t_run


# ── Result aggregation ────────────────────────────────────────────────────────
def load_results_from_disk(save_dir: str) -> dict[int, dict]:
    """
    Read every available timings.npz under *save_dir* and return a mapping
    ``{n_quadpoints: {'t_jit': ..., 't_run': ..., <result arrays>}}``.
    Only entries whose npz file exists are included.
    """
    results: dict[int, dict] = {}
    for n in N_QUADPOINTS_LIST:
        timing_file = Path(save_dir) / f'n_{n}' / 'timings.npz'
        if timing_file.exists():
            saved = np.load(timing_file)
            results[n] = {
                't_jit': float(saved['t_jit']),
                't_run': float(saved['t_run']),
                **{k: jnp.array(v) for k, v in saved.items()
                   if k not in ('t_jit', 't_run')},
            }
    return results


"""validate_with_dolfinx.py — Independent validation of CoilFEM forward solve.

Runs TWO parallel dolfinx linear-elasticity solves per base coil, sharing the
exact same mesh, material, gravity, and Winkler spring BC as CoilFEM, but
using different sources for the Lorentz body force:

  Run A — full-stack validation
      B_self and B_ext are recomputed from scratch via volumetric Biot-Savart
      over the existing coil meshes.  No analytical formula (Landreman /
      simsopt) and no extra air-domain mesh are needed; the integral kernel is
      (mu0/4pi) * J_vol × r / |r|³ summed over all source cells.  The
      contribution from the same cell as the target (singular) is skipped and
      documented as an O(h) error.  dolfinx solves the linear-elasticity
      problem with this independently computed body force.

  Run B — elasticity-only validation
      B_self and B_ext are taken directly from ``coil_fem.run()`` so the body
      force is bit-identical to what JAX-FEM consumed.  dolfinx solves the
      same elasticity problem.  Residual differences vs. CoilFEM isolate the
      elasticity solver (jax-fem vs. dolfinx), not the EM model.

Together the two runs let you attribute any disagreement with CoilFEM to
either the EM model (Run A vs. Run B) or the elasticity implementation
(Run B vs. CoilFEM).

Outputs
-------
``<out_dir>/runA_volumetric_BS/coil_{i:02d}_run_dolfinx.vtu``
``<out_dir>/runA_volumetric_BS/dolfinx_run.npz``
``<out_dir>/runB_coilfem_B/coil_{i:02d}_run_dolfinx.vtu``
``<out_dir>/runB_coilfem_B/dolfinx_run.npz``

When ``compare=True`` is passed to :func:`validate_with_dolfinx`, also writes:

``<out_dir>/comparison_summary.csv``

Usage
-----
::

    from validate_with_dolfinx import validate_with_dolfinx
    results = validate_with_dolfinx(coil_fem, out_dir="dfx_out", compare=True)

Dependencies
------------
dolfinx is not in pyproject.toml.  Install with::

    mamba install -n rod -c conda-forge fenics-dolfinx mpich pyvista

or the petsc4py/slepc4py equivalents for your platform.  mpi4py and meshio are
also required (meshio is already in the [fem] extra).
"""


# --------------------------------------------------------------------------- #
MU0_OVER_4PI: float = 1e-7  # [T·m/A]
# --------------------------------------------------------------------------- #


# ============================================================================
# Dolfinx availability guard
# ============================================================================

def _require_dolfinx():
    """Import dolfinx; raise ImportError with install instructions if missing."""
    try:
        import dolfinx        # noqa: F401
        from mpi4py import MPI # noqa: F401
    except ImportError as e:
        raise ImportError(
            "dolfinx / mpi4py not found.\n"
            "Install with:\n"
            "  mamba install -n rod -c conda-forge fenics-dolfinx mpich\n"
            f"(original error: {e})"
        ) from e


# ============================================================================
# 1. Per-base-coil geometry at FEM quadrature points
# ============================================================================

def _get_base_coil_data(coil_fem: "CoilFEM") -> list[dict]:
    """Evaluate physical mesh points, quad positions, JxW, and J direction for
    every base coil at the FEM quadrature points.

    Returns a list of n_base dicts, each containing:

    ``pts``       (n_nodes, 3)       physical mesh node positions [m]
    ``cells``     (n_cells, 4 or 10) TET4/TET10 connectivity (int32)
    ``pqp``       (n_cells, n_q, 3) physical quadrature-point positions [m]
    ``JxW``       (n_cells, n_q)    Jacobian × quad weight [m³]
    ``t_hat_q``   (n_cells, n_q, 3) unit tangent at each quad point
    ``A``         float             cross-section area [m²]
    ``n_cells``   int
    ``n_quads``   int
    """
    import interpax
    import jax.numpy as jnp

    from coilforce.curve_jax import CurveXYZFourierJAX
    from coilforce.elasticity import recompute_fe_geometry

    base_curves_dofs = [c.dofs for c in coil_fem.base_curves_jax]
    out = []

    for i in range(len(coil_fem.base_curves_jax)):
        base = coil_fem.base_curves_jax[i]
        dofs_i = base_curves_dofs[i]
        meta = coil_fem._grid_meta[i]
        prob_i = coil_fem._problems[i]

        # Physical mesh-node positions (differentiable, but we convert to numpy)
        pts_i = coil_fem._mesh_points_from_dofs(dofs_i, i)

        # FE geometry: JxW and physical quad points
        _, JxW_i, _, pqp_i = recompute_fe_geometry(
            pts_i,
            prob_i._cells_jnp,
            prob_i._sg_ref,
            prob_i._sv,
            prob_i._qw,
        )

        # Unit tangent at FEM quad points (same interpolation as _body_force_at_quads)
        curve = CurveXYZFourierJAX(base.quadpoints, dofs_i, base.order)
        gammadash_cl = curve.gammadash()           # (n_phi, 3)
        t_hat_cl = gammadash_cl / jnp.linalg.norm(
            gammadash_cl, axis=1, keepdims=True
        )
        phi_q = meta["phi_quad"]                   # (n_cells, n_quads) static
        n_cells = meta["n_cells"]
        n_quads = meta["n_quads"]
        t_hat_q = interpax.interp1d(
            phi_q.ravel(),
            curve.quadpoints,
            t_hat_cl,
            method="cubic2",
            period=1.0,
        ).reshape(n_cells, n_quads, 3)

        out.append(
            {
                "pts":     np.asarray(pts_i,       dtype=np.float64),
                "cells":   np.asarray(prob_i._cells_jnp, dtype=np.int32),
                "pqp":     np.asarray(pqp_i,       dtype=np.float64),
                "JxW":     np.asarray(JxW_i,       dtype=np.float64),
                "t_hat_q": np.asarray(t_hat_q,     dtype=np.float64),
                "A":       float(meta["cross_section_area"]),
                "n_cells": n_cells,
                "n_quads": n_quads,
            }
        )

    return out


# ============================================================================
# 2. Symmetry expansion of source-tet data
# ============================================================================

def _build_symmetry_sources(
    coil_fem: "CoilFEM",
    base_data: list[dict],
) -> list[dict]:
    """Expand the base-coil source data to all symmetry images.

    Expansion order mirrors ``src/coilforce/symmetries.py``::

        for k in 0..nfp-1:
            for flip in [False, True] if stellsym else [False]:
                for i in 0..n_base-1:
                    ...

    Returns a list of n_total dicts, each with:

    ``pqp``           (n_cells, n_q, 3)  transformed quad point positions [m]
    ``J_vol``         (n_cells, n_q, 3)  J * JxW [A·m]  (amps × metres)
    ``base_coil_idx`` int                which base coil this image comes from
    ``current``       float              signed current [A]
    """
    from coilforce.symmetries import apply_symmetries_to_currents

    n_base  = len(coil_fem.base_curves_jax)
    nfp     = coil_fem.nfp
    stellsym = coil_fem.stellsym

    all_currents = np.asarray(
        apply_symmetries_to_currents(coil_fem.base_currents_jax, nfp, stellsym)
    )  # (n_total,)

    sources: list[dict] = []
    j = 0
    for k in range(nfp):
        phi = 2.0 * np.pi * k / nfp
        c, s = np.cos(phi), np.sin(phi)
        # Rotation about z-axis (same convention as symmetries.py _rotate_points_z)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        flip_list = [False, True] if stellsym else [False]
        for flip in flip_list:
            # Stellarator flip: negate y and z (applied AFTER rotation)
            F = np.diag([1.0, -1.0, -1.0]) if flip else np.eye(3)
            T = F @ R  # combined transform; T @ x for a column vector
            for i in range(n_base):
                bd   = base_data[i]
                I_j  = float(all_currents[j])
                A_i  = bd["A"]

                # Transform quad-point positions  (n_cells, n_q, 3) @ T.T
                pqp_j = bd["pqp"] @ T.T

                # Transform unit tangent (same linear transform as positions)
                t_hat_j = bd["t_hat_q"] @ T.T

                # J_vol = (I/A) * t_hat * JxW  [A/m² × m³ = A·m]
                J_vol = (I_j / A_i) * t_hat_j * bd["JxW"][:, :, None]

                sources.append(
                    {
                        "pqp":           pqp_j,
                        "J_vol":         J_vol,
                        "base_coil_idx": i,
                        "current":       I_j,
                    }
                )
                j += 1

    return sources


# ============================================================================
# 3. Volumetric Biot-Savart kernel
# ============================================================================

def _biot_savart_volumetric(
    target_pqp: np.ndarray,
    sources: list[dict],
    *,
    self_src_idx: int | None = None,
    chunk_cells: int = 64,
) -> np.ndarray:
    """Compute B at target quad points via volumetric Biot-Savart.

    Kernel::

        B(x_t) = (mu0/4pi) * Σ_src  J_vol_src × r_ts / |r_ts|³

    where ``r_ts = x_t - x_src`` and ``J_vol = J * dV`` [A·m].

    Parameters
    ----------
    target_pqp : (n_tgt_cells, n_q, 3)
        Physical positions of target FEM quadrature points.
    sources : list of dicts
        Each dict must have keys ``pqp`` and ``J_vol``; see
        :func:`_build_symmetry_sources`.
    self_src_idx : int or None
        If provided, skip contributions from source cells whose cell index
        equals the target cell index (same-cell singularity avoidance for
        B_self).  Only the source at this list position is subjected to the
        skip; all other sources contribute fully.
    chunk_cells : int
        Number of *target cells* to process in one numpy batch.  Increase for
        speed (but higher memory), decrease if RAM is limited.

    Returns
    -------
    np.ndarray, (n_tgt_cells, n_q, 3), dtype float64, [T]
    """
    n_tgt_cells, n_quads, _ = target_pqp.shape
    N_target = n_tgt_cells * n_quads
    target_flat = target_pqp.reshape(N_target, 3)  # (N_target, 3)

    B_flat = np.zeros((N_target, 3), dtype=np.float64)

    for j, src in enumerate(sources):
        src_pts  = src["pqp"].reshape(-1, 3)   # (N_src, 3)
        src_Jvol = src["J_vol"].reshape(-1, 3)  # (N_src, 3)
        n_src_cells  = src["pqp"].shape[0]
        n_src_quads  = src["pqp"].shape[1]
        N_src = src_pts.shape[0]
        is_self = (j == self_src_idx)

        # Process target points in chunks to limit peak memory usage.
        # Memory per chunk: chunk_cells * n_quads * N_src * 3 * 8 bytes
        chunk_flat = chunk_cells * n_quads
        for i_start in range(0, N_target, chunk_flat):
            i_end = min(i_start + chunk_flat, N_target)
            x_chunk = target_flat[i_start:i_end]  # (chunk, 3)

            # r[t, s] = x_t - x_src_s
            r = x_chunk[:, None, :] - src_pts[None, :, :]  # (chunk, N_src, 3)
            r_norm = np.linalg.norm(r, axis=-1)             # (chunk, N_src)

            # Validity mask: exclude near-zero distances (degenerate / same point)
            valid = r_norm > 1e-20  # (chunk, N_src)

            if is_self:
                # Skip contributions from the same source cell as the target.
                # Flat target index t → target cell index: t // n_quads
                # Flat source index s → source cell index: s // n_src_quads
                tgt_cell = (
                    (np.arange(i_start, i_end) // n_quads)[:, None]
                )  # (chunk, 1)
                src_cell = (
                    (np.arange(N_src) // n_src_quads)[None, :]
                )  # (1, N_src)
                same_cell = tgt_cell == src_cell  # (chunk, N_src)
                valid = valid & ~same_cell

            r_norm3 = np.where(valid, r_norm ** 3, 1.0)  # avoid division by 0

            # Cross product: J_vol_src × r_ts  → (chunk, N_src, 3)
            cross = np.cross(src_Jvol[None, :, :], r)

            B_flat[i_start:i_end] += MU0_OVER_4PI * np.sum(
                cross * (valid / r_norm3)[:, :, None],
                axis=1,
            )

    return B_flat.reshape(n_tgt_cells, n_quads, 3)


def _compute_b_fields_volumetric(
    coil_fem: "CoilFEM",
    base_data: list[dict],
    sources: list[dict],
    chunk_cells: int = 64,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute B_self and B_ext for all base coils via volumetric Biot-Savart.

    B_self[i] — field from coil i's own 3D volume elements on coil i's quad
                 points (same-cell contributions skipped).
    B_ext[i]  — field from every other symmetry image (all j ≠ i where i is
                 the source at expansion index 0, i.e. k=0, flip=False, coil i).

    This matches the split in ``CoilFEM._body_force_at_quads``:
    ``B_ext`` zeros ``all_currents[coil_idx]`` which corresponds to expansion
    index ``coil_idx`` == i for k=0, flip=False.

    Returns
    -------
    B_self_list : list[ndarray(n_cells_i, n_q, 3)]  [T]
    B_ext_list  : list[ndarray(n_cells_i, n_q, 3)]  [T]
    """
    n_base = len(coil_fem.base_curves_jax)
    B_self_list: list[np.ndarray] = []
    B_ext_list:  list[np.ndarray] = []

    for i in range(n_base):
        target_pqp = base_data[i]["pqp"]  # (n_cells, n_q, 3)
        n_c = base_data[i]["n_cells"]
        n_q = base_data[i]["n_quads"]

        # self_src_idx = i because in the expansion order, expansion index i
        # corresponds to k=0, flip=False, base coil i (the first n_base entries).
        print(
            f"  [vol-BS] coil {i}: B_self "
            f"({n_c} target cells × {n_q} quad pts)..."
        )
        B_self_i = _biot_savart_volumetric(
            target_pqp,
            [sources[i]],    # only the k=0, flip=False image of coil i
            self_src_idx=0,  # skip same-cell within that single source
            chunk_cells=chunk_cells,
        )

        print(f"  [vol-BS] coil {i}: B_ext ...")
        # All sources EXCEPT expansion index i
        sources_ext = [src for j, src in enumerate(sources) if j != i]
        B_ext_i = _biot_savart_volumetric(
            target_pqp,
            sources_ext,
            self_src_idx=None,  # no singularity: target coil is absent
            chunk_cells=chunk_cells,
        )

        B_self_list.append(B_self_i)
        B_ext_list.append(B_ext_i)

    return B_self_list, B_ext_list


# ============================================================================
# 4. Body-force assembly (same as CoilFEM, minus the Landreman B_self formula)
# ============================================================================

def _build_fvol(
    J_q:       np.ndarray,  # (n_cells, n_q, 3) [A/m²]
    B_self_q:  np.ndarray,  # (n_cells, n_q, 3) [T]
    B_ext_q:   np.ndarray,  # (n_cells, n_q, 3) [T]
    rho:       float,       # density [kg/m³], 0 if gravity off
    g_vec:     np.ndarray,  # (3,)  [m/s²]
) -> np.ndarray:
    """Lorentz + gravity body force at FEM quad points.

    Returns
    -------
    np.ndarray, (n_cells, n_q, 3), [N/m³]
    """
    f = np.cross(J_q, B_self_q + B_ext_q)
    if rho != 0.0:
        f = f + rho * g_vec[None, None, :]
    return f


# ============================================================================
# 5. Dolfinx linear-elasticity solve (single coil)
# ============================================================================

def _dolfinx_solve(
    pts_np:          np.ndarray,  # (n_nodes, 3) [m]
    cells_np:        np.ndarray,  # (n_cells, 4 or 10) int TET4/TET10 connectivity
    fvol_q:          np.ndarray,  # (n_cells, n_quads, 3) [N/m³] per-Gauss-pt body force
    spring_k_nodes:  np.ndarray,  # (n_nodes,) [N/m³] spring stiffness (0 = free)
    E:               float,       # Young's modulus [Pa]
    nu:              float,       # Poisson's ratio
    eps_th:          np.ndarray | None = None,  # (3, 3) thermal eigenstrain or None
) -> dict:
    """Solve linear elasticity with Winkler BC in dolfinx.

    Body force is injected via a Quadrature-degree function space so that each
    Gauss point within a cell carries its own force value.  This matches
    CoilFEM's per-quadrature-point load-vector assembly exactly::

        L(Nᵢ) = Σ_q  f(xq) · Nᵢ(xq) · JxWq

    Both JAX-FEM (via jax_fem.basis) and dolfinx (via basix) use the same
    basix quadrature rule, so the within-cell Gauss-point ordering is
    identical; only a cell-level permutation (coilforce → dolfinx) is needed.

    Von Mises stress is evaluated at the same Gauss points and averaged per
    cell, matching CoilFEM's ``mean(VM(xq), q=1..n_quads)`` convention as
    written by ``save_run_vtu``.

    For TET10 elements, P2 Lagrange basis with a 4-point volume rule
    (``quadrature_degree=2``) and a 6-point Dunavant face rule
    (``quadrature_degree=4``) matches CoilFEM's ``gauss_order=2`` volume
    and ``gauss_order=4`` face settings (the latter overrides the JAX-FEM
    default to ensure full-rank Winkler surface stiffness on TRI6 faces).

    Thermal eigenstrain ``eps_th`` (if not None) is applied in **both** the
    constitutive law (weak-form bilinear part) and the von Mises post-processing,
    matching ``coilforce.thermal.cauchy_stress_with_thermal_strain``.

    Parameters
    ----------
    fvol_q : (n_cells, n_quads, 3)
        Body force at every FEM quadrature point [N/m³], in coilforce cell
        order.  For TET10 with ``gauss_order=2`` this has ``n_quads=4``.
    spring_k_nodes : (n_nodes,)
        Winkler spring stiffness at every mesh node [N/m³].  Non-zero only on
        the Winkler surface; interior nodes must be 0.  Applied as a
        Pn-interpolated surface spring on all exterior facets via the bilinear
        form, where n matches the displacement basis degree.

    Returns
    -------
    dict with keys:
        ``displacement``   (n_nodes, 3) [m]  — nodal displacement in coilforce
                           node order (KD-tree matched from dolfinx DOF order).
        ``von_mises_cell`` (n_cells,)   [Pa] — mean von Mises stress over
                           Gauss points per cell, in coilforce cell order.
    """
    _require_dolfinx()

    import dolfinx
    import dolfinx.mesh as dfx_mesh
    import basix.ufl
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import LinearProblem
    from mpi4py import MPI
    from scipy.spatial import cKDTree

    # Detect element order: TET10 → P2, TET4 → P1.
    is_tet10 = cells_np.shape[1] == 10
    fem_degree = 2 if is_tet10 else 1

    # ── Build dolfinx mesh ────────────────────────────────────────────────────
    # TET4  → degree-1 straight-sided geometry (4 corner nodes / cell).
    # TET10 → degree-2 ISOPARAMETRIC (curved-sided) geometry using all 10 nodes,
    #         so the dolfinx geometry matches coilforce's curved TET10 elements.
    #         The curved-edge mesh update moved the midside nodes off the
    #         straight chord midpoints; coilforce's JAX-FEM solve uses those
    #         positions in its isoparametric element map, so a straight
    #         P2-on-degree-1 reference would solve a slightly different
    #         (chord-sided) problem and would also break the P2 DOF coordinate
    #         matching below (dolfinx would place edge DOFs at chord midpoints).
    #
    # coilforce/meshio store TET10 connectivity in VTK ordering, but dolfinx
    # `create_mesh` expects basix/DefElement ordering.  `perm_vtk` maps
    # VTK → dolfinx via  a_dolfin[i] = a_vtk[p[i]]  (i.e. cells[:, p]).
    corner_cells = cells_np[:, :4] if is_tet10 else cells_np
    geo_degree = 2 if is_tet10 else 1
    el_def = basix.ufl.element("Lagrange", "tetrahedron", geo_degree, shape=(3,))
    ufl_domain = ufl.Mesh(el_def)
    if is_tet10:
        from dolfinx.mesh import CellType as _CellType
        try:                                       # dolfinx >= ~0.8
            from dolfinx.io.utils import cell_perm_vtk as _perm_vtk
        except ImportError:                        # older builds expose it on cpp
            from dolfinx.cpp.io import perm_vtk as _perm_vtk
        vtk_to_dolfinx = np.asarray(
            _perm_vtk(_CellType.tetrahedron, 10), dtype=np.int32
        )
        cells_in = cells_np[:, vtk_to_dolfinx].astype(np.int64)
    else:
        cells_in = cells_np.astype(np.int64)
    mesh = dfx_mesh.create_mesh(
        MPI.COMM_WORLD,
        cells_in,
        e=ufl_domain,
        x=pts_np.astype(np.float64),
    )

    # ── Build coilforce → dolfinx cell-index permutation ──────────────────────
    # `create_mesh` reorders cells for cache locality, so DG0 dof index c does
    # NOT correspond to coilforce cell c.  Match cell centroids via KD-tree to
    # build the permutation in both directions; same trick as for Pn nodes
    # below.  Required for:
    #   * Injecting cell-mean body force into DG0 functions in the right slot.
    #   * Returning von Mises (DG0) in coilforce cell order to the caller.
    # `compute_midpoints` averages each cell's *vertex* (corner) geometry nodes
    # (its source assumes a linear geometry), so it returns the corner centroid
    # even for the degree-2 isoparametric mesh.  For curved TET10 the mean of
    # all 10 nodes no longer equals the corner centroid, so match against the
    # corner-only centroid.  As a guard against dolfinx versions that average
    # all geometry nodes instead, fall back to the all-node centroid.
    tdim = mesh.topology.dim
    n_cells_dfx = mesh.topology.index_map(tdim).size_local
    cell_idx_dfx = np.arange(n_cells_dfx, dtype=np.int32)
    cent_dfx = dfx_mesh.compute_midpoints(mesh, tdim, cell_idx_dfx)
    if cent_dfx.shape[0] != cells_np.shape[0]:
        raise RuntimeError(
            f"Cell count mismatch: dolfinx={cent_dfx.shape[0]}, "
            f"coilforce={cells_np.shape[0]}.  Did dolfinx drop cells?"
        )
    tol = 1e-8 * (np.abs(pts_np).max() + 1.0)
    tree_cell = cKDTree(cent_dfx)
    best = None
    for cent_cf in (pts_np[corner_cells].mean(axis=1),   # corner centroid
                    pts_np[cells_np].mean(axis=1)):       # all-node centroid
        dist_c, mapping = tree_cell.query(cent_cf)        # (n_cells,)
        if best is None or np.max(dist_c) < best[0]:
            best = (np.max(dist_c), mapping)
    dist_c_max, dolfinx_for_coilforce = best
    if dist_c_max > tol:
        raise RuntimeError(
            f"Cell centroid match failed (max distance "
            f"{dist_c_max:.3e} m > tol {tol:.3e}).  Dolfinx mesh may "
            "differ from input topology."
        )
    if len(np.unique(dolfinx_for_coilforce)) != n_cells_dfx:
        raise RuntimeError("Non-unique cell-centroid match; KD-tree mapping is degenerate.")

    # ── Material ─────────────────────────────────────────────────────────────
    lam_val = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_val  = E / (2.0 * (1.0 + nu))
    lam = fem.Constant(mesh, lam_val)
    mu  = fem.Constant(mesh, mu_val)

    # ── Thermal eigenstrain ──────────────────────────────────────────────────
    if eps_th is not None:
        eps_th_ufl = ufl.as_tensor(eps_th.tolist())
    else:
        eps_th_ufl = None

    # ── Function spaces ───────────────────────────────────────────────────────
    V = fem.functionspace(mesh, ("Lagrange", fem_degree, (3,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # ── Constitutive law (with optional thermal eigenstrain) ─────────────────
    # ``sigma`` is the full thermoelastic stress σ = C:(ε − ε_th).  It is used
    # only for the von Mises post-processing, where the argument is a known
    # Function.  It must NOT be used in the bilinear form: when ε_th is a
    # non-None constant, the ``−C:ε_th`` term carries no trial-function
    # argument, so placing it in ``a`` mixes arity-2 and arity-1 terms and
    # makes FFCx raise ``ArityMismatch: Adding expressions with non-matching
    # form arguments () vs ('v_1',)``.  The thermal term is instead moved to
    # the load vector L (see below).
    def sigma(u_fn):
        eps_u = ufl.sym(ufl.grad(u_fn))
        eps_m = eps_u - eps_th_ufl if eps_th_ufl is not None else eps_u
        return lam * ufl.tr(eps_m) * ufl.Identity(3) + 2 * mu * eps_m

    def sigma_elastic(u_fn):
        """Thermal-free stress C:ε(u) for the (arity-2) bilinear form."""
        eps_u = ufl.sym(ufl.grad(u_fn))
        return lam * ufl.tr(eps_u) * ufl.Identity(3) + 2 * mu * eps_u

    # ── Volume integration measure ────────────────────────────────────────────
    # vol_quad_degree must match JAX-FEM's gauss_order:
    #   TET10 → gauss_order=2 → 4 Gauss pts → quadrature_degree=2
    #   TET4  → gauss_order=1 → 1 Gauss pt  → quadrature_degree=1
    # The explicit degree is required so that the Quadrature function space
    # used for the body force and von Mises is evaluated at the same points
    # as the dx measure (dolfinx raises an error if they disagree).
    vol_quad_degree = 2 if is_tet10 else 1
    vol_meta = {"quadrature_degree": vol_quad_degree}
    dx = ufl.dx(metadata=vol_meta)

    # ── Quadrature element (version-safe) ─────────────────────────────────────
    # basix 0.7+ accepts keyword args; basix 0.6 uses a different positional
    # signature.  The string shortcut ("Quadrature", n) is only registered in
    # dolfinx >= 0.9, so we always use basix.ufl.quadrature_element directly.
    try:
        _vol_quad_el = basix.ufl.quadrature_element(
            "tetrahedron", value_shape=(), degree=vol_quad_degree
        )
    except TypeError:
        # basix 0.6 API: quadrature_element(cell, scheme, degree, value_shape)
        _vol_quad_el = basix.ufl.quadrature_element(
            "tetrahedron", "default", vol_quad_degree, ()
        )

    # ── Bilinear form ─────────────────────────────────────────────────────────
    # Purely elastic (thermal-free) stress so both factors carry (u, v): arity 2.
    a = ufl.inner(sigma_elastic(u), ufl.sym(ufl.grad(v))) * dx

    # ── Body force (Quadrature space, one value per Gauss point) ─────────────
    # Use a Quadrature-degree function space so each of the n_quads Gauss
    # points per cell carries its own body force value.  This makes the load
    # vector assembly identical to CoilFEM:
    #
    #     L(Nᵢ) = Σ_q  f(xq) · Nᵢ(xq) · JxWq
    #
    # rather than the cruder DG0 approximation:
    #
    #     L(Nᵢ) = f̄_cell · ∫ Nᵢ dV
    #
    # which differs for TET10 (P2 basis) when f varies within the cell.
    #
    # DOF layout of a scalar Quadrature-k space: for each dolfinx cell (in
    # dolfinx order), n_quads consecutive DOFs hold the per-point values in
    # basix quadrature ordering.  Both JAX-FEM and dolfinx derive their
    # quadrature rules from basix, so the within-cell ordering is the same;
    # only a cell-level permutation is required.
    #
    # Build inverse permutation: coilforce_for_dolfinx[c_dfx] = c_cf
    coilforce_for_dolfinx = np.empty(n_cells_dfx, dtype=np.intp)
    coilforce_for_dolfinx[dolfinx_for_coilforce] = np.arange(n_cells_dfx, dtype=np.intp)
    # Reorder fvol_q from coilforce cell order to dolfinx cell order.
    fvol_q_dfx = fvol_q[coilforce_for_dolfinx]          # (n_cells_dfx, n_quads, 3)
    QS = fem.functionspace(mesh, _vol_quad_el)
    fx_fn = fem.Function(QS); fx_fn.x.array[:] = fvol_q_dfx[:, :, 0].ravel(); fx_fn.x.scatter_forward()
    fy_fn = fem.Function(QS); fy_fn.x.array[:] = fvol_q_dfx[:, :, 1].ravel(); fy_fn.x.scatter_forward()
    fz_fn = fem.Function(QS); fz_fn.x.array[:] = fvol_q_dfx[:, :, 2].ravel(); fz_fn.x.scatter_forward()
    f_ufl = ufl.as_vector([fx_fn, fy_fn, fz_fn])

    L = ufl.inner(f_ufl, v) * dx

    # Thermal eigenstrain contributes an equivalent load.  Splitting
    # σ = C:(ε(u) − ε_th) in the weak form ∫ σ:ε(v) dV = ∫ f·v dV gives
    #     ∫ C:ε(u):ε(v) dV = ∫ f·v dV + ∫ C:ε_th:ε(v) dV,
    # so the constant thermal stress C:ε_th enters the RHS (arity 1) rather
    # than the bilinear form.  This keeps ``a`` purely arity-2 and matches
    # CoilFEM, where ε_th acts through the constitutive law as a thermal load.
    if eps_th_ufl is not None:
        sigma_th = lam * ufl.tr(eps_th_ufl) * ufl.Identity(3) + 2 * mu * eps_th_ufl
        L = L + ufl.inner(sigma_th, ufl.sym(ufl.grad(v))) * dx

    # ── Winkler spring foundation (all exterior facets) ───────────────────────
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    ext_facets = dfx_mesh.exterior_facet_indices(mesh.topology)

    if np.any(spring_k_nodes > 0):
        V_s = fem.functionspace(mesh, ("Lagrange", fem_degree))
        k_fn = fem.Function(V_s)

        # Map coilforce node index → dolfinx DOF index via coordinate matching.
        # For TET10, V_s has P2 DOFs at corner nodes AND edge midpoints, which
        # correspond exactly to all nodes in pts_np.
        dof_coords_s = V_s.tabulate_dof_coordinates()  # (n_dofs, 3)
        tree_dof = cKDTree(dof_coords_s)
        dist, dof_for_node = tree_dof.query(pts_np)    # (n_nodes,)
        if np.max(dist) > 1e-8 * (np.abs(pts_np).max() + 1.0):
            raise RuntimeError(
                f"P{fem_degree} DOF coords deviate from mesh nodes by up to "
                f"{np.max(dist):.3e} m.  Dolfinx may have reordered nodes; "
                "coordinate matching failed."
            )

        k_fn.x.array[:] = 0.0
        k_fn.x.array[dof_for_node] = spring_k_nodes
        k_fn.x.scatter_forward()

        facet_tag_vals = np.ones(len(ext_facets), dtype=np.int32)
        facet_tags = dfx_mesh.meshtags(mesh, fdim, ext_facets, facet_tag_vals)
        # Match CoilFEM's Winkler face quadrature.
        # elasticity.py:custom_init() overrides the JAX-FEM default to gauss_order=4
        # for TET10 so the local Winkler stiffness  K_ij = ∫ k N_i N_j dS  has
        # full rank on TRI6 faces (n_face=6 DOFs needs ≥6 quad points).
        # The JAX-FEM default gauss_order=2 gives only 3 face quad points → rank-3
        # deficiency → artificially soft Winkler springs and ~50% von Mises error.
        #
        # Matching degrees (both backed by basix.make_quadrature):
        #   TET4  → quadrature_degree=2 → 3-point on TRI3  (full rank, P1 face = 3 DOFs)
        #   TET10 → quadrature_degree=4 → 6-point Dunavant on TRI6 (full rank, P2 face = 6 DOFs)
        face_quad_deg = 4 if is_tet10 else 2
        ds = ufl.Measure(
            "ds", domain=mesh,
            subdomain_data=facet_tags, subdomain_id=1,
            metadata={"quadrature_degree": face_quad_deg},
        )
        a = a + k_fn * ufl.inner(u, v) * ds

    # ── Solve ─────────────────────────────────────────────────────────────────
    # Try MUMPS (direct, robust); fall back to default iterative solver
    try:
        time0 = time.time()
        problem = LinearProblem(
            a, L, bcs=[],
            petsc_options_prefix="dolfinx_validate",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type":  "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        time1 = time.time()
        uh = problem.solve()
        time2 = time.time()
        print(f"  [dolfinx] construction time: {time1 - time0:.3f}s")
        print(f"  [dolfinx] solve time:        {time2 - time1:.3f}s")
        print(f"  [dolfinx] total time:        {time2 - time0:.3f}s")
    except TypeError:
        problem = LinearProblem(a, L, bcs=[], petsc_options_prefix="dolfinx_fallback")
        uh = problem.solve()

    # ── Extract displacement in coilforce node order ──────────────────────────
    # Use a scalar Pn space to get DOF coordinates (same degree as displacement).
    # For TET10, P2 DOF positions include corner nodes and edge midpoints, which
    # matches all nodes in pts_np; the KD-tree maps each coilforce node to the
    # corresponding dolfinx DOF index.  uh.x.array is interleaved (x,y,z) per
    # DOF for both P1 and P2 vector spaces, so reshape(-1, 3) is always correct.
    V_s2 = fem.functionspace(mesh, ("Lagrange", fem_degree))
    dof_coords2 = V_s2.tabulate_dof_coordinates()   # (n_dofs, 3)
    tree_dof2 = cKDTree(dof_coords2)
    _, dof_for_node2 = tree_dof2.query(pts_np)       # coilforce_node → dolfinx_dof

    uh_arr = uh.x.array.reshape(-1, 3)               # (n_dolfinx_dofs, 3)
    displacement = np.zeros((pts_np.shape[0], 3), dtype=np.float64)
    displacement = uh_arr[dof_for_node2]             # (n_nodes, 3), coilforce order

    # ── Von Mises stress (Gauss-point evaluation, cell mean) ──────────────────
    # Evaluate von Mises at the same n_quads Gauss points used for FEM
    # assembly, then average per cell.  This matches CoilFEM's convention:
    #
    #     vm_cell[c] = mean( VM(xq),  q = 1..n_quads )
    #
    # as written by save_run_vtu:
    #     jnp.mean(result['von_mises'][i], axis=-1)
    #
    # Using a DG0 projection (VM at centroid) instead would differ because
    # von Mises is a nonlinear function of strain: mean(sqrt(…)) ≠ sqrt(…)
    # evaluated at the centroid.
    def von_mises_ufl(u_fn):
        sig = sigma(u_fn)
        s   = sig - (ufl.tr(sig) / 3.0) * ufl.Identity(3)
        return ufl.sqrt(1.5 * ufl.inner(s, s) + 1e-30)

    W = fem.functionspace(mesh, _vol_quad_el)
    interp_pts = W.element.interpolation_points
    if callable(interp_pts):
        interp_pts = interp_pts()
    vm_expr = fem.Expression(von_mises_ufl(uh), interp_pts)
    vm_fn = fem.Function(W)
    vm_fn.interpolate(vm_expr)
    # vm_fn.x.array: (n_cells_dfx * n_quads,) in dolfinx cell order, basix quad order
    n_quads_vm = vm_fn.x.array.shape[0] // n_cells_dfx
    vm_per_quad_dfx = vm_fn.x.array.reshape(n_cells_dfx, n_quads_vm)  # (n_cells_dfx, n_quads)

    # Permute from dolfinx to coilforce cell order and average over Gauss pts.
    von_mises_cell = np.asarray(
        vm_per_quad_dfx[dolfinx_for_coilforce].mean(axis=1), dtype=np.float64
    ).copy()

    return {
        "displacement":   displacement,   # (n_nodes, 3) [m]
        "von_mises_cell": von_mises_cell, # (n_cells,) [Pa] — coilforce cell order
    }


# ============================================================================
# 6. Spring stiffness node array (coilforce → numpy full-node array)
# ============================================================================

def _spring_k_node_array(
    coil_fem: "CoilFEM",
    i: int,
    pts_i: np.ndarray,
    dofs_i,
    support_dofs_i,
) -> np.ndarray:
    """Return spring stiffness at every mesh node for base coil ``i``.

    Nodes not on the Winkler surface carry stiffness 0.

    Returns
    -------
    np.ndarray, (n_nodes,) [N/m³]
    """
    if not coil_fem.base_support_fns:
        return np.zeros(pts_i.shape[0], dtype=np.float64)

    winkler_k = float(coil_fem.problem_options["winkler_k"])
    surf_idx  = np.asarray(coil_fem._surface_node_indices[i], dtype=np.int32)
    weights   = np.asarray(
        coil_fem._compute_support_weights(i, pts_i, dofs_i, support_dofs_i),
        dtype=np.float64,
    )

    k_nodes = np.zeros(pts_i.shape[0], dtype=np.float64)
    k_nodes[surf_idx] = weights * winkler_k
    return k_nodes


# ============================================================================
# 7. Output helpers
# ============================================================================

def _write_vtu(
    path: str,
    pts_np:         np.ndarray,   # (n_nodes, 3)
    cells_np:       np.ndarray,   # (n_cells, 4)
    displacement:   np.ndarray,   # (n_nodes, 3)
    von_mises_cell: np.ndarray,   # (n_cells,)  [Pa]
    B_self_cell:    np.ndarray,   # (n_cells, 3) [T]
    B_ext_cell:     np.ndarray,   # (n_cells, 3) [T]
    f_vol_cell:     np.ndarray,   # (n_cells, 3) [N/m³]
    spring_k_nodes: np.ndarray,   # (n_nodes,) [N/m³]
    support_weights: np.ndarray,  # (n_nodes,)
) -> None:
    """Write a single coil result VTU with the same field names as save_run_vtu."""
    import meshio

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    vm_mpa = von_mises_cell / 1e6  # Pa → MPa

    point_data: dict = {"displacement_m": displacement}
    if np.any(spring_k_nodes > 0):
        point_data["support_weights"] = support_weights
        point_data["spring_k_Npm3"]  = spring_k_nodes

    cell_data: dict = {
        "von_mises_MPa": [vm_mpa],
        "f_vol_Npm3":    [f_vol_cell],
        "B_self_T":      [B_self_cell],
        "B_ext_T":       [B_ext_cell],
    }

    meshio_cell_type = "tetra10" if cells_np.shape[1] == 10 else "tetra"
    meshio.Mesh(
        points=pts_np,
        cells=[(meshio_cell_type, cells_np.astype(np.int32))],
        point_data=point_data,
        cell_data=cell_data,
    ).write(path)


def _save_npz(path: str, results: dict) -> None:
    """Save the dolfinx results dict in a format mirroring CoilFEM.run().

    Each list entry for base coil ``i`` is stored under key ``{field}_{i}``.
    Load back with::

        d = np.load(path)
        B_self = [d[f"B_self_{i}"] for i in range(n_base)]
    """
    flat: dict = {}
    for key, lst in results.items():
        if lst is None:
            continue
        if isinstance(lst, (list, tuple)):
            for i, arr in enumerate(lst):
                if arr is not None:
                    flat[f"{key}_{i}"] = np.asarray(arr)
        else:
            flat[key] = np.asarray(lst)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **flat)


# ============================================================================
# 8. Per-run pipeline (shared by Run A and Run B)
# ============================================================================

def _run_dolfinx_pipeline(
    coil_fem:      "CoilFEM",
    base_data:     list[dict],
    B_self_list:   list[np.ndarray],  # (n_cells_i, n_q, 3) [T]  per base coil
    B_ext_list:    list[np.ndarray],  # (n_cells_i, n_q, 3) [T]  per base coil
    out_subdir:    str,
    label:         str = "dolfinx",
) -> dict:
    """Run dolfinx elasticity per base coil and write outputs to ``out_subdir``.

    Returns a dict with the same keys as ``CoilFEM.run()``:
    ``displacements``, ``von_mises``, ``mesh_points``, ``support_weights``,
    ``f_vol``, ``B_self``, ``B_ext``, ``solutions`` (set to None).
    """
    os.makedirs(out_subdir, exist_ok=True)

    n_base = len(coil_fem.base_curves_jax)
    base_curves_dofs = [c.dofs for c in coil_fem.base_curves_jax]
    # _validate_support_dofs(None, n_base) returns [None] * n_base
    support_dofs: list = [None] * n_base

    # Material
    E   = coil_fem._E
    nu  = coil_fem._nu
    rho = coil_fem._rho

    # Thermal eigenstrain: ε_th = -itc · I (integral thermal contraction).
    eps_th: np.ndarray | None = None
    if coil_fem._itc is not None:
        itc = coil_fem._itc
        eps_th = -np.eye(3, dtype=np.float64) * itc
        print(f"  Thermal eigenstrain active: itc={itc}, eps_th[0,0]={eps_th[0,0]:.4e}")

    # Gravity
    if coil_fem.gravity_options is not None:
        grav_rho = float(coil_fem.gravity_options.get("density", rho))
        g_vec    = np.asarray(
            coil_fem.gravity_options.get("g_vec", (0.0, 0.0, -9.80665)),
            dtype=np.float64,
        )
    else:
        grav_rho = 0.0
        g_vec    = np.zeros(3, dtype=np.float64)

    # Collect results
    disp_list:    list = []
    vm_list:      list = []
    pts_list:     list = []
    wt_list:      list = []
    fvol_list:    list = []
    Bself_out:    list = []
    Bext_out:     list = []

    for i in range(n_base):
        print(f"\n[{label}] base coil {i} ...")
        bd      = base_data[i]
        pts_np  = bd["pts"]
        cells_np= bd["cells"]
        n_cells = bd["n_cells"]
        n_quads = bd["n_quads"]
        A_i     = bd["A"]
        I_i     = float(coil_fem.base_currents_jax[i])
        dofs_i  = base_curves_dofs[i]

        # J at quad points: (I/A) * t_hat_q
        J_q = (I_i / A_i) * bd["t_hat_q"]  # (n_cells, n_quads, 3)

        # Body force at quad points
        f_vol_q = _build_fvol(
            J_q,
            B_self_list[i],
            B_ext_list[i],
            grav_rho,
            g_vec,
        )  # (n_cells, n_quads, 3)

        # Cell-mean body force — kept for VTU visualization output only.
        # The dolfinx solve now receives the full per-Gauss-point array.
        f_vol_cell_mean = f_vol_q.mean(axis=1)  # (n_cells, 3)

        # Spring stiffness at mesh nodes
        import jax.numpy as jnp
        spring_k_nodes = _spring_k_node_array(
            coil_fem, i, jnp.asarray(pts_np), dofs_i, support_dofs[i]
        )
        spring_k_nodes_np = np.asarray(spring_k_nodes)

        # Support weights (unscaled) for VTU output
        if coil_fem.base_support_fns:
            winkler_k   = float(coil_fem.problem_options["winkler_k"])
            support_weights_np = spring_k_nodes_np / (winkler_k + 1e-300)
        else:
            support_weights_np = np.zeros(pts_np.shape[0], dtype=np.float64)

        # ── dolfinx solve ────────────────────────────────────────────────────
        t0     = time.perf_counter()
        sol = _dolfinx_solve(
            pts_np,
            cells_np,
            f_vol_q,          # full per-Gauss-point body force (n_cells, n_quads, 3)
            spring_k_nodes_np,
            E,
            nu,
            eps_th,
        )
        t_run  = time.perf_counter() - t0
        
        timing_file = Path(out_subdir) / 'timings.npz'
        np.savez(
            timing_file,
            t_run=t_run,
        )
        displacement   = sol["displacement"]    # (n_nodes, 3) [m]
        vm_cell        = sol["von_mises_cell"]  # (n_cells,) [Pa]

        # Cell-mean B fields for VTU / npz output
        B_self_cell = B_self_list[i].mean(axis=1)  # (n_cells, 3)
        B_ext_cell  = B_ext_list[i].mean(axis=1)   # (n_cells, 3)

        # ── Write VTU ────────────────────────────────────────────────────────
        vtu_path = os.path.join(out_subdir, f"coil_{i:02d}_run_dolfinx.vtu")
        _write_vtu(
            vtu_path,
            pts_np,
            cells_np,
            displacement,
            vm_cell,
            B_self_cell,
            B_ext_cell,
            f_vol_cell_mean,
            spring_k_nodes_np,
            support_weights_np,
        )
        print(f"  → wrote {vtu_path}")

        # ── Accumulate results ───────────────────────────────────────────────
        # Broadcast cell-mean von Mises to (n_cells, n_quads) for format compat
        vm_broadcast = np.tile(vm_cell[:, None], (1, n_quads))  # (n_cells, n_q)

        disp_list.append(displacement)
        vm_list.append(vm_broadcast)
        pts_list.append(pts_np)
        wt_list.append(support_weights_np)
        fvol_list.append(f_vol_q)           # (n_cells, n_q, 3) — quad resolution
        Bself_out.append(B_self_list[i])    # (n_cells, n_q, 3)
        Bext_out.append(B_ext_list[i])      # (n_cells, n_q, 3)

    results = {
        "solutions":       None,         # not available from dolfinx
        "displacements":   disp_list,
        "von_mises":       vm_list,
        "mesh_points":     pts_list,
        "support_weights": wt_list,
        "f_vol":           fvol_list,
        "B_self":          Bself_out,
        "B_ext":           Bext_out,
    }

    npz_path = os.path.join(out_subdir, "dolfinx_run.npz")
    _save_npz(npz_path, results)
    print(f"\n  → wrote {npz_path}")

    return results


# ============================================================================
# 9. Optional comparison against CoilFEM.run()
# ============================================================================

def _compare_results(
    coil_fem:       "CoilFEM",
    base_data:      list[dict],
    coilfem_result: dict,
    runA_result:    dict | None,
    runB_result:    dict | None,
    out_csv:        str,
) -> None:
    """Build a three-way comparison summary CSV.

    Compares at the same physical locations:
    - Nodal displacement:  KD-tree match on node coordinates.
    - Cell von Mises:      KD-tree match on cell centroids.
    - Cell B_self / B_ext / f_vol:  same cell centroids.

    CSV columns are grouped as::

        RunA vs CoilFEM | RunB vs CoilFEM | RunA vs RunB

    Each group reports mean and max relative error for displacement magnitude,
    von Mises stress, B_self magnitude, B_ext magnitude, and f_vol magnitude.
    """
    from scipy.spatial import cKDTree

    run_pairs = [("A", "CF"), ("B", "CF"), ("A", "B")]

    def _match_nodes(src_pts, src_vals, tgt_pts):
        _, idx = cKDTree(src_pts).query(tgt_pts)
        return src_vals[idx]

    def _match_cells(src_c, src_vals, tgt_c):
        _, idx = cKDTree(src_c).query(tgt_c)
        return src_vals[idx]

    def _rel_err(a, b, eps=1e-12):
        return float(np.mean(np.abs(a - b) / (np.abs(b) + eps)))

    def _max_rel_err(a, b, eps=1e-12):
        return float(np.max(np.abs(a - b) / (np.abs(b) + eps)))

    def _extract(run_result, coil_idx):
        """Extract and reshape per-coil fields from a run result dict."""
        if run_result is None:
            return None
        cells_i = base_data[coil_idx]["cells"]  # shared connectivity
        r_disp  = np.asarray(run_result["displacements"][coil_idx])
        r_vm    = np.asarray(run_result["von_mises"][coil_idx]).mean(-1)
        r_Bs    = np.asarray(run_result["B_self"][coil_idx]).mean(1)
        r_Be    = np.asarray(run_result["B_ext"][coil_idx]).mean(1)
        r_fv    = np.asarray(run_result["f_vol"][coil_idx]).mean(1)
        r_pts   = np.asarray(run_result["mesh_points"][coil_idx])
        r_c     = r_pts[cells_i].mean(1)
        return dict(disp=r_disp, vm=r_vm, Bs=r_Bs, Be=r_Be, fv=r_fv,
                    pts=r_pts, centroids=r_c)

    rows: list[dict] = []
    n_base = len(coil_fem.base_curves_jax)

    for i in range(n_base):
        bd      = base_data[i]
        pts_np  = bd["pts"]
        cells_i = bd["cells"]

        # Cell centroids (same for all runs since mesh is identical)
        centroids = pts_np[cells_i].mean(axis=1)  # (n_cells, 3)

        # CoilFEM reference values
        cf_pts = np.asarray(coilfem_result["mesh_points"][i])
        cf_c   = cf_pts[cells_i].mean(1)
        cf_disp = np.asarray(coilfem_result["displacements"][i])
        cf_vm   = np.asarray(coilfem_result["von_mises"][i]).mean(-1)
        cf_Bs   = np.asarray(coilfem_result["B_self"][i]).mean(1)
        cf_Be   = np.asarray(coilfem_result["B_ext"][i]).mean(1)
        cf_fv   = np.asarray(coilfem_result["f_vol"][i]).mean(1)

        A_vals = _extract(runA_result, i)
        B_vals = _extract(runB_result, i)

        row: dict = {"coil": i}

        for (tag_a, tag_b) in run_pairs:
            r_a = {"A": A_vals, "B": B_vals}.get(tag_a)
            if r_a is None:
                continue

            if tag_b == "CF":
                b_pts = cf_pts;  b_c = cf_c
                r_b_disp = cf_disp
                r_b_vm   = cf_vm
                r_b_Bs   = np.linalg.norm(cf_Bs, axis=-1)
                r_b_Be   = np.linalg.norm(cf_Be, axis=-1)
                r_b_fv   = np.linalg.norm(cf_fv, axis=-1)
            else:  # tag_b == "B"
                if B_vals is None:
                    for suf in ["disp_mean_rel", "disp_max_rel",
                                "vm_mean_rel",   "vm_max_rel",
                                "Bs_mean_rel",   "Bs_max_rel",
                                "Be_mean_rel",   "Be_max_rel",
                                "fv_mean_rel",   "fv_max_rel"]:
                        row[f"{tag_a}_{tag_b}_{suf}"] = float("nan")
                    continue
                b_pts = B_vals["pts"];  b_c = B_vals["centroids"]
                r_b_disp = B_vals["disp"]
                r_b_vm   = B_vals["vm"]
                r_b_Bs   = np.linalg.norm(B_vals["Bs"], axis=-1)
                r_b_Be   = np.linalg.norm(B_vals["Be"], axis=-1)
                r_b_fv   = np.linalg.norm(B_vals["fv"], axis=-1)

            disp_a     = _match_nodes(r_a["pts"], r_a["disp"], b_pts)
            disp_a_mag = np.linalg.norm(disp_a, axis=-1)
            disp_b_mag = np.linalg.norm(r_b_disp, axis=-1)

            vm_a = _match_cells(r_a["centroids"], r_a["vm"], b_c)
            Bs_a = np.linalg.norm(_match_cells(r_a["centroids"], r_a["Bs"], b_c), axis=-1)
            Be_a = np.linalg.norm(_match_cells(r_a["centroids"], r_a["Be"], b_c), axis=-1)
            fv_a = np.linalg.norm(_match_cells(r_a["centroids"], r_a["fv"], b_c), axis=-1)

            p = f"{tag_a}_{tag_b}"
            row[f"{p}_disp_mean_rel"] = _rel_err(disp_a_mag, disp_b_mag)
            row[f"{p}_disp_max_rel"]  = _max_rel_err(disp_a_mag, disp_b_mag)
            row[f"{p}_vm_mean_rel"]   = _rel_err(vm_a, r_b_vm)
            row[f"{p}_vm_max_rel"]    = _max_rel_err(vm_a, r_b_vm)
            row[f"{p}_Bs_mean_rel"]   = _rel_err(Bs_a, r_b_Bs)
            row[f"{p}_Bs_max_rel"]    = _max_rel_err(Bs_a, r_b_Bs)
            row[f"{p}_Be_mean_rel"]   = _rel_err(Be_a, r_b_Be)
            row[f"{p}_Be_max_rel"]    = _max_rel_err(Be_a, r_b_Be)
            row[f"{p}_fv_mean_rel"]   = _rel_err(fv_a, r_b_fv)
            row[f"{p}_fv_max_rel"]    = _max_rel_err(fv_a, r_b_fv)

        rows.append(row)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if not rows:
        print("[compare] No results to compare.")
        return

    import csv
    all_keys = ["coil"]
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[compare] wrote {out_csv}")

    # ── Print human-readable table ────────────────────────────────────────────
    print(
        f"\n{'coil':>5}  "
        + "  ".join(
            f"{'A/CF_vm_max%':>13} {'B/CF_vm_max%':>13} {'A/B_vm_max%':>12}"
            .split()
        )
    )
    for row in rows:
        print(
            f"{row['coil']:>5}  "
            f"  {row.get('A_CF_vm_max_rel', float('nan'))*100:12.2f}%"
            f"  {row.get('B_CF_vm_max_rel', float('nan'))*100:12.2f}%"
            f"  {row.get('A_B_vm_max_rel',  float('nan'))*100:11.2f}%"
        )


# ============================================================================
# 10. Public entry point
# ============================================================================

def validate_with_dolfinx(
    coil_fem: "CoilFEM",
    out_dir:  str = "dolfinx_out",
    *,
    run_A:              bool  = True,
    run_B:              bool  = True,
    result_coilfem            = None,
    bs_chunk_cells:     int   = 64,
) -> dict:
    """Run the dolfinx validation for all base coils.

    Parameters
    ----------
    coil_fem : CoilFEM
        A fully-constructed :class:`coilforce.coil_fem.CoilFEM` object.
        The DOFs / currents used are the initial ones stored in the object.
    out_dir : str
        Root output directory.  Sub-folders ``runA_volumetric_BS/`` and
        ``runB_coilfem_B/`` are created automatically.
    run_A : bool
        If True, compute B_self / B_ext via volumetric Biot-Savart and run
        the dolfinx elasticity solve (full-stack validation).
    run_B : bool
        If True, take B_self / B_ext from ``coil_fem.run()`` and run the
        same dolfinx elasticity solve (elasticity-only validation).
    compare : bool
        If True, also call ``coil_fem.run()`` (or reuse the result from Run B
        if already computed), compare results cell-by-cell, and write
        ``<out_dir>/comparison_summary.csv``.
    bs_chunk_cells : int
        Number of target cells per Biot-Savart chunk (memory vs. speed trade-off).

    Returns
    -------
    dict with keys ``"runA"``, ``"runB"``, ``"coilfem"`` (each is the
    respective results dict or ``None`` if not computed).
    """
    _require_dolfinx()

    if not coil_fem.base_support_fns:
        import warnings
        warnings.warn(
            "CoilFEM has no support functions — no Winkler BC will be applied in "
            "dolfinx.  The system may be singular unless the body force alone "
            "determines the deformation uniquely (e.g. self-equilibrated load).  "
            "If JAX-FEM uses Dirichlet BCs you must add them here manually.",
            stacklevel=2,
        )

    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: compute base-coil geometric data ─────────────────────────────
    print("=" * 60)
    print("Building per-base-coil FEM quadrature data ...")
    base_data = _get_base_coil_data(coil_fem)

    # ── Step 2: symmetry expansion of source tets ────────────────────────────
    print("Expanding to all symmetry images ...")
    sources = _build_symmetry_sources(coil_fem, base_data)
    n_total = len(sources)
    n_base  = len(coil_fem.base_curves_jax)
    print(
        f"  n_base={n_base}, nfp={coil_fem.nfp}, "
        f"stellsym={coil_fem.stellsym} → n_total={n_total}"
    )

    result_A      = None
    result_B      = None

    # ── Run A: volumetric Biot-Savart ─────────────────────────────────────────
    if run_A:
        print("\n" + "=" * 60)
        print("Run A: volumetric Biot-Savart + dolfinx elasticity")
        print("=" * 60)
        print("Computing B_self and B_ext via volumetric Biot-Savart ...")
        B_self_A, B_ext_A = _compute_b_fields_volumetric(
            coil_fem, base_data, sources, chunk_cells=bs_chunk_cells
        )
        subdir_A = os.path.join(out_dir, "runA_volumetric_BS")
        result_A = _run_dolfinx_pipeline(
            coil_fem, base_data, B_self_A, B_ext_A, subdir_A, label="Run A"
        )

    # ── Run B: CoilFEM B fields → dolfinx elasticity only ────────────────────
    if run_B or result_coilfem:
        print("\n" + "=" * 60)
        print("Run B")
        print("=" * 60)
        print("  coil_fem.run() finished.")

    if run_B:
        B_self_B = [
            np.asarray(result_coilfem["B_self"][i]) for i in range(n_base)
        ]
        B_ext_B  = [
            np.asarray(result_coilfem["B_ext"][i])  for i in range(n_base)
        ]
        subdir_B = os.path.join(out_dir, "runB_coilfem_B")
        result_B = _run_dolfinx_pipeline(
            coil_fem, base_data, B_self_B, B_ext_B, subdir_B, label="Run B"
        )

    # ── Compare ───────────────────────────────────────────────────────────────
    if result_coilfem:
        print("\n" + "=" * 60)
        print("Comparing results ...")
        _compare_results(
            coil_fem,
            base_data,
            result_coilfem,
            result_A,
            result_B,
            out_csv=os.path.join(out_dir, "comparison_summary.csv"),
        )

    print("\nDone.  Output directory:", out_dir)
    return {"runA": result_A, "runB": result_B, "coilfem": result_coilfem}


# ============================================================================
# CLI / demo entry point
# ============================================================================

if __name__ == "__main__":
    print(
        "validate_with_dolfinx.py\n"
        "Usage:\n"
        "  from validate_with_dolfinx import validate_with_dolfinx\n"
        "  results = validate_with_dolfinx(coil_fem, out_dir='dfx_out', compare=True)\n"
        "\n"
        "The CoilFEM object must be constructed before calling this function.\n"
        "See docs/tutorial/optimize_von_mises.ipynb for an example setup."
    )
