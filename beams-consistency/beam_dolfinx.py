"""Import the fused full-body mesh and run dolfinx elasticity with quadpoint force.

Reads ``full_mesh.msh`` (TET10) and the classification/clamp sidecar
``body_force.npz`` (no ``f_vol``).  Lorentz body force is evaluated at volume
quadrature points via coil-fem Biot–Savart / ``B_self_quadrature`` (same
physics as CoilFEM), packed into a dolfinx ``quadrature_element`` Function,
and assembled with a matching ``quadrature_degree`` on ``dx``.

Winkler weights use analytic clamp spheres at facet quadrature points
(``clamp_sigmoid``), not a P2-interpolated nodal weight field.

No thermal eigenstrain is applied; CoilFEM's notebook case may include
``itc = 0.0029`` prestress.
"""

from __future__ import annotations

import time

import basix
import basix.ufl
import interpax
import jax.numpy as jnp
import meshio
import numpy as np
import ufl
from mpi4py import MPI
from scipy.spatial import cKDTree
from simsopt import load

from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx import plot as dfx_plot
from dolfinx.fem.petsc import LinearProblem

try:
    from dolfinx.io import gmshio  # dolfinx <= 0.9
except ImportError:  # dolfinx >= 0.10 renamed gmshio -> gmsh
    from dolfinx.io import gmsh as gmshio

from coil_fem.magnetic import biot_savart, B_self_quadrature, lorentz_body_force
from coil_fem.utils import clamp_sigmoid

MESH_PATH = "full_mesh.msh"
BODY_FORCE_PATH = "body_force.npz"
JSTRESS_PATH = "Jstress.json"
OUT_VTU = "full_body_elasticity.vtu"
VOL_QUAD_DEG = 4
CHUNK = 32_768
N_PHI = 4096
UV_TOL = 2e-2

# VTK cell-type integers returned by dolfinx.plot.vtk_mesh → meshio names.
_VTK_TO_MESHIO = {
    10: "tetra",                 # VTK_TETRA
    24: "tetra10",               # VTK_QUADRATIC_TETRA
    71: "tetra10",               # VTK_LAGRANGE_TETRAHEDRON (order 2 → 10 nodes)
}


def import_mesh(path: str = MESH_PATH):
    """Read the fused device mesh into a dolfinx ``Mesh``."""
    result = gmshio.read_from_msh(
        path, MPI.COMM_WORLD, rank=0, gdim=3,
    )
    if hasattr(result, "mesh"):  # dolfinx >= 0.10 returns a MeshData object
        domain = result.mesh
        cell_tags = getattr(result, "cell_tags", None)
        facet_tags = getattr(result, "facet_tags", None)
    else:  # dolfinx <= 0.9 returns a (mesh, cell_tags, facet_tags) tuple
        domain, cell_tags, facet_tags = result
    return domain, cell_tags, facet_tags


def exterior_node_indices(domain):
    """Return DOF indices and coordinates of every exterior mesh node."""
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    exterior_facets = dmesh.exterior_facet_indices(domain.topology)

    degree = domain.geometry.cmap.degree
    V = fem.functionspace(domain, ("Lagrange", degree))
    exterior_dofs = fem.locate_dofs_topological(V, fdim, exterior_facets)
    exterior_coords = V.tabulate_dof_coordinates()[exterior_dofs]
    return exterior_dofs, exterior_coords


def clamp_weight_at(
    points: np.ndarray,
    centers: np.ndarray,
    r_clamp: float,
    eps_sigmoid: float,
) -> np.ndarray:
    """Sum of ``clamp_sigmoid`` over clamp centres at each query point."""
    pts = jnp.asarray(points, dtype=jnp.float64)
    ctr = jnp.asarray(centers, dtype=jnp.float64)
    d_sq = jnp.sum((pts[:, None, :] - ctr[None, :, :]) ** 2, axis=-1)
    return np.asarray(
        clamp_sigmoid(d_sq, float(r_clamp), float(eps_sigmoid)).sum(axis=-1),
        dtype=np.float64,
    )


def clamp_weight_ufl(x, centers, r_clamp: float, eps_sigmoid: float):
    """UFL weight matching :func:`clamp_weight_at` at facet quadrature points."""
    width2 = float(eps_sigmoid * r_clamp) ** 2
    r2 = float(r_clamp) ** 2
    w = 0
    for c in np.asarray(centers, dtype=np.float64):
        d_sq = (
            (x[0] - float(c[0])) ** 2
            + (x[1] - float(c[1])) ** 2
            + (x[2] - float(c[2])) ** 2
        )
        w = w + 1.0 / (1.0 + ufl.exp(-(r2 - d_sq) / width2))
    return w


def load_mesh_sidecar(path: str = BODY_FORCE_PATH):
    """Load classification / clamp / material sidecar (no ``f_vol`` / ``B_*``)."""
    data = np.load(path)
    required = (
        "clamp_centers", "r_clamp", "eps_sigmoid", "k_clamp",
        "E", "nu", "rho", "g_vec",
    )
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(
            f"{path} missing keys {missing}. Re-run mesh.ipynb Step 4 export."
        )
    if "f_vol" in data.files:
        raise ValueError(
            f"{path} still contains 'f_vol'. Re-export with the stripped "
            f"mesh.ipynb (force is computed here at volume quads)."
        )
    return data


def physical_volume_quad_points(domain, vol_quad_deg: int = VOL_QUAD_DEG):
    """Physical volume quadrature coordinates, shape ``(n_cells, n_quads, 3)``.

    Uses a vector ``quadrature_element`` Function interpolated from
    ``SpatialCoordinate``, so ordering matches FFCx assembly for the same
    ``quadrature_degree``.
    """
    Qe = basix.ufl.quadrature_element(
        domain.topology.cell_name(),
        value_shape=(3,),
        degree=vol_quad_deg,
    )
    W = fem.functionspace(domain, Qe)
    x = ufl.SpatialCoordinate(domain)
    ipts = W.element.interpolation_points
    if callable(ipts):
        ipts = ipts()
    coords_f = fem.Function(W)
    coords_f.interpolate(fem.Expression(x, ipts))

    tdim = domain.topology.dim
    n_cells = domain.topology.index_map(tdim).size_local
    arr = np.asarray(coords_f.x.array, dtype=np.float64).reshape(-1, 3)
    n_quads = arr.shape[0] // n_cells
    if arr.shape[0] != n_cells * n_quads:
        raise RuntimeError(
            f"quad packing mismatch: {arr.shape[0]} values, "
            f"{n_cells} cells"
        )
    return arr.reshape(n_cells, n_quads, 3), n_quads


def _symmetry_Q_list(nfp: int, stellsym: bool) -> np.ndarray:
    flip_list = (False, True) if stellsym else (False,)
    flip_Q = np.diag([1.0, -1.0, -1.0])
    Q_list = []
    for k in range(nfp):
        phi_k = 2.0 * np.pi * k / nfp
        c, s = np.cos(phi_k), np.sin(phi_k)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for flip in flip_list:
            Q_list.append(flip_Q @ rot if flip else rot)
    return np.asarray(Q_list)


def inverse_map_points(points: np.ndarray, Jstress, *, w1: float, w2: float):
    """Classify physical points into conductor vs beam via (φ, u, v) pullback.

    Same algorithm as mesh.ipynb Step 4A, applied to an arbitrary point cloud
    (volume quadrature points here).

    Returns
    -------
    owner_coil, owner_sym : (N,) int8
    phi, u, v : (N,) float64
    """
    support = Jstress.fem.support
    nfp, stellsym = support.nfp, support.stellsym
    n_base = len(Jstress.fem.meshes)
    Q_list = _symmetry_Q_list(nfp, stellsym)
    n_sym = Q_list.shape[0]

    phi_s = np.linspace(0.0, 1.0, N_PHI, endpoint=False)
    dist_cut = 0.5 * np.hypot(w1, w2) * 1.05

    gamma_s, p_s, q_s = [], [], []
    for coil_mesh in Jstress.fem.meshes:
        fc = coil_mesh.framed_curve
        g = np.asarray(fc.curve.gamma_eval(phi_s))
        _, p, q = fc.rotated_frame_eval(phi_s)
        gamma_s.append(np.asarray(g))
        p_s.append(np.asarray(p))
        q_s.append(np.asarray(q))

    sample_pts, sample_lab = [], []
    for s in range(n_sym):
        Q = Q_list[s]
        for i in range(n_base):
            sample_pts.append(gamma_s[i] @ Q.T)
            labs = np.empty((N_PHI, 3), dtype=np.int32)
            labs[:, 0] = s
            labs[:, 1] = i
            labs[:, 2] = np.arange(N_PHI)
            sample_lab.append(labs)
    tree = cKDTree(np.vstack(sample_pts))
    sample_lab = np.vstack(sample_lab)

    X = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    N = X.shape[0]
    dist, nn = tree.query(X, k=3)
    cand_lab = sample_lab[nn]

    owner_sym = np.full(N, -1, dtype=np.int8)
    owner_coil = np.full(N, -1, dtype=np.int8)
    phi_out = np.zeros(N, dtype=np.float64)
    u_out = np.zeros(N, dtype=np.float64)
    v_out = np.zeros(N, dtype=np.float64)

    for cand in range(3):
        unset = owner_coil < 0
        if not unset.any():
            break
        idxs = np.where(unset)[0]
        s_c = cand_lab[idxs, cand, 0]
        i_c = cand_lab[idxs, cand, 1]
        phi0 = phi_s[cand_lab[idxs, cand, 2]]
        d_c = dist[idxs, cand]
        keep_mask = d_c <= dist_cut
        for i in range(n_base):
            m = keep_mask & (i_c == i)
            if not m.any():
                continue
            sel = idxs[m]
            s_sel = s_c[m]
            phi_guess = phi0[m].astype(np.float64)
            Qs = Q_list[s_sel]
            y = np.einsum("ni,nij->nj", X[sel], Qs)

            fc = Jstress.fem.meshes[i].framed_curve
            curve = fc.curve
            phi_n = phi_guess.copy()
            for _ in range(2):
                g0 = np.asarray(curve.gamma_eval(phi_n, 0))
                g1 = np.asarray(curve.gamma_eval(phi_n, 1))
                g2 = np.asarray(curve.gamma_eval(phi_n, 2))
                dvec = y - g0
                g = np.sum(dvec * g1, axis=1)
                gp = -np.sum(g1 * g1, axis=1) + np.sum(dvec * g2, axis=1)
                phi_n = phi_n - g / np.where(np.abs(gp) < 1e-30, 1e-30, gp)
            phi_n = np.mod(phi_n, 1.0)

            p_i = np.asarray(interpax.interp1d(
                phi_n, phi_s, p_s[i], method="cubic2", period=1.0,
            ))
            q_i = np.asarray(interpax.interp1d(
                phi_n, phi_s, q_s[i], method="cubic2", period=1.0,
            ))
            g0 = np.asarray(curve.gamma_eval(phi_n, 0))
            dvec = y - g0
            u = 2.0 * np.sum(dvec * p_i, axis=1) / w1
            v = 2.0 * np.sum(dvec * q_i, axis=1) / w2
            inside = (np.abs(u) <= 1.0 + UV_TOL) & (np.abs(v) <= 1.0 + UV_TOL)

            take = sel[inside]
            owner_sym[take] = s_sel[inside].astype(np.int8)
            owner_coil[take] = i
            phi_out[take] = phi_n[inside]
            u_out[take] = u[inside]
            v_out[take] = v[inside]

    return owner_coil, owner_sym, phi_out, u_out, v_out, Q_list


def compute_f_vol_at_quads(
    quad_pts: np.ndarray,
    Jstress,
    *,
    rho: float,
    g_vec: np.ndarray,
    w1: float,
    w2: float,
) -> np.ndarray:
    """Lorentz (+ gravity) body force at physical volume quads.

    Parameters
    ----------
    quad_pts : ndarray, shape (n_cells, n_quads, 3)
    Jstress : CoilFEMObjective
    rho, g_vec, w1, w2

    Returns
    -------
    ndarray, shape (n_cells, n_quads, 3)
    """
    n_cells, n_quads, _ = quad_pts.shape
    pts = quad_pts.reshape(-1, 3)
    owner_coil, owner_sym, phi_out, u_out, v_out, Q_list = inverse_map_points(
        pts, Jstress, w1=w1, w2=w2,
    )
    n_cond = int((owner_coil >= 0).sum())
    print(
        f"quad classification: {n_cond} / {pts.shape[0]} conductor "
        f"({100 * n_cond / pts.shape[0]:.1f}%)"
    )

    base_curves_dofs = [c.dofs for c in Jstress.fem.base_curves_jax]
    all_gammas, all_gammadashs, all_currents = Jstress.fem._expand_geometry(
        base_curves_dofs, Jstress.fem.base_currents_jax,
    )
    n_base = len(Jstress.fem.meshes)

    f_vol = np.zeros((pts.shape[0], 3), dtype=np.float64)
    y_base = np.zeros_like(pts)
    cond = owner_coil >= 0
    if cond.any():
        Qs = Q_list[owner_sym[cond]]
        y_base[cond] = np.einsum("ni,nij->nj", pts[cond], Qs)

    def _chunked_biot(targets, currents):
        out = np.empty((targets.shape[0], 3), dtype=np.float64)
        tgt = jnp.asarray(targets)
        cur = jnp.asarray(currents)
        for lo in range(0, targets.shape[0], CHUNK):
            hi = min(lo + CHUNK, targets.shape[0])
            out[lo:hi] = np.asarray(biot_savart(
                tgt[lo:hi], all_gammas, all_gammadashs, cur,
            ))
        return out

    def _chunked_B_self(fc, I, cross_section, phi_arr, uv_arr):
        out = np.empty((phi_arr.shape[0], 3), dtype=np.float64)
        for lo in range(0, phi_arr.shape[0], CHUNK):
            hi = min(lo + CHUNK, phi_arr.shape[0])
            phi_b = jnp.asarray(phi_arr[lo:hi])[:, None]
            uv_b = jnp.asarray(uv_arr[lo:hi])[:, None, :]
            out[lo:hi] = np.asarray(
                B_self_quadrature(fc, I, cross_section, phi_b, uv_b)[:, 0]
            )
        return out

    t0 = time.time()
    for i in range(n_base):
        sel = np.where(owner_coil == i)[0]
        if sel.size == 0:
            continue
        coil_mesh = Jstress.fem.meshes[i]
        fc = coil_mesh.framed_curve
        curve = fc.curve
        A = coil_mesh.cross_section_area
        I = all_currents[i]
        cross_section = {"shape": "rect", "w1": coil_mesh.w1, "w2": coil_mesh.w2}

        phi_i = phi_out[sel]
        uv_i = np.stack([u_out[sel], v_out[sel]], axis=1)
        y_i = y_base[sel]

        gd = np.asarray(curve.gamma_eval(phi_i, 1))
        t_hat = gd / np.linalg.norm(gd, axis=1, keepdims=True)
        J = (float(I) / A) * t_hat

        B_self = _chunked_B_self(fc, I, cross_section, phi_i, uv_i)
        B_ext = _chunked_biot(y_i, all_currents.at[i].set(0.0))
        f_base = np.asarray(lorentz_body_force(
            jnp.asarray(J), jnp.asarray(B_self + B_ext),
        ))
        Q_n = Q_list[owner_sym[sel]]
        f_vol[sel] = np.einsum("nij,nj->ni", Q_n, f_base)
        print(f"  coil {i}: {sel.size} quads")

    f_vol += float(rho) * np.asarray(g_vec, dtype=np.float64)[None, :]
    print(
        f"body force at quads in {time.time() - t0:.1f} s, "
        f"|f|_max = {np.linalg.norm(f_vol, axis=1).max():.3e} N/m³"
    )
    return f_vol.reshape(n_cells, n_quads, 3)


def make_quadrature_force_function(domain, f_vol_q: np.ndarray, vol_quad_deg: int):
    """Pack ``(n_cells, n_quads, 3)`` into a vector quadrature-element Function."""
    Qe = basix.ufl.quadrature_element(
        domain.topology.cell_name(),
        value_shape=(3,),
        degree=vol_quad_deg,
    )
    W = fem.functionspace(domain, Qe)
    f = fem.Function(W)
    flat = np.asarray(f_vol_q, dtype=np.float64).reshape(-1)
    if flat.size != f.x.array.size:
        raise RuntimeError(
            f"f_vol packing mismatch: got {flat.size}, "
            f"quadrature space has {f.x.array.size}"
        )
    f.x.array[:] = flat
    return f


def solve_elasticity_winkler(
    domain,
    f,
    *,
    E: float,
    nu: float,
    k_clamp: float,
    clamp_centers: np.ndarray,
    r_clamp: float,
    eps_sigmoid: float,
    vol_quad_deg: int = VOL_QUAD_DEG,
):
    """Solve linear elasticity with analytic clamp Winkler + quadpoint body force."""
    lam_val = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_val = E / (2.0 * (1.0 + nu))
    lam = fem.Constant(domain, lam_val)
    mu = fem.Constant(domain, mu_val)

    degree = domain.geometry.cmap.degree
    V = fem.functionspace(domain, ("Lagrange", degree, (3,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def sigma_elastic(u_fn):
        eps = ufl.sym(ufl.grad(u_fn))
        return lam * ufl.tr(eps) * ufl.Identity(3) + 2 * mu * eps

    def sigma(u_fn):
        return sigma_elastic(u_fn)

    dx = ufl.Measure(
        "dx", domain=domain,
        metadata={"quadrature_degree": vol_quad_deg},
    )
    a = ufl.inner(sigma_elastic(u), ufl.sym(ufl.grad(v))) * dx
    L = ufl.inner(f, v) * dx

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    ext_facets = dmesh.exterior_facet_indices(domain.topology)
    facet_tags = dmesh.meshtags(
        domain, fdim, ext_facets, np.ones(len(ext_facets), dtype=np.int32),
    )
    face_quad_deg = 4 if degree == 2 else 2
    ds = ufl.Measure(
        "ds", domain=domain,
        subdomain_data=facet_tags, subdomain_id=1,
        metadata={"quadrature_degree": face_quad_deg},
    )
    x = ufl.SpatialCoordinate(domain)
    w_ufl = clamp_weight_ufl(x, clamp_centers, r_clamp, eps_sigmoid)
    a = a + float(k_clamp) * w_ufl * ufl.inner(u, v) * ds

    try:
        problem = LinearProblem(
            a, L, bcs=[],
            petsc_options_prefix="full_body_elasticity",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        uh = problem.solve()
    except TypeError:
        problem = LinearProblem(
            a, L, bcs=[], petsc_options_prefix="full_body_elasticity_fb",
        )
        uh = problem.solve()

    # Von Mises at volume Gauss points, then cell mean.
    vm_quad_degree = 2 if degree == 2 else 1
    try:
        vol_quad_el = basix.ufl.quadrature_element(
            "tetrahedron", value_shape=(), degree=vm_quad_degree,
        )
    except TypeError:
        vol_quad_el = basix.ufl.quadrature_element(
            "tetrahedron", "default", vm_quad_degree, (),
        )

    def von_mises_ufl(u_fn):
        sig = sigma(u_fn)
        s = sig - (ufl.tr(sig) / 3.0) * ufl.Identity(3)
        return ufl.sqrt(1.5 * ufl.inner(s, s) + 1e-30)

    W = fem.functionspace(domain, vol_quad_el)
    interp_pts = W.element.interpolation_points
    if callable(interp_pts):
        interp_pts = interp_pts()
    vm_expr = fem.Expression(von_mises_ufl(uh), interp_pts)
    vm_fn = fem.Function(W)
    vm_fn.interpolate(vm_expr)

    n_cells = domain.topology.index_map(tdim).size_local
    n_quads = vm_fn.x.array.shape[0] // n_cells
    von_mises_Pa = np.asarray(
        vm_fn.x.array.reshape(n_cells, n_quads).mean(axis=1),
        dtype=np.float64,
    ).copy()

    return {"uh": uh, "von_mises_Pa": von_mises_Pa}


def write_result_vtu(
    domain,
    out_path: str,
    *,
    uh,
    von_mises_Pa: np.ndarray,
    clamp_centers: np.ndarray,
    r_clamp: float,
    eps_sigmoid: float,
    k_clamp: float,
    f_vol_cell: np.ndarray | None = None,
    atol: float = 1e-9,
) -> None:
    """Write VTU: point displacement/w_clamp; cell von Mises and cell-mean f_vol."""
    topology, cell_types, x = dfx_plot.vtk_mesh(domain)
    vtk_type = int(cell_types[0])
    if vtk_type not in _VTK_TO_MESHIO:
        raise ValueError(
            f"Unsupported VTK cell type {vtk_type}; "
            f"expected one of {sorted(_VTK_TO_MESHIO)}"
        )
    meshio_type = _VTK_TO_MESHIO[vtk_type]
    n_per_cell = int(topology[0])
    cells = topology.reshape(-1, n_per_cell + 1)[:, 1:]

    degree = domain.geometry.cmap.degree
    V_s = fem.functionspace(domain, ("Lagrange", degree))
    dist, idx = cKDTree(V_s.tabulate_dof_coordinates()).query(x, k=1)
    if float(np.max(dist)) > atol:
        raise RuntimeError(
            f"VTK geometry ↔ DOF coordinate join failed: max dist "
            f"{float(np.max(dist)):.3e} m > atol {atol}"
        )

    w_nodes = clamp_weight_at(
        np.asarray(x, dtype=np.float64), clamp_centers, r_clamp, eps_sigmoid,
    )
    point_data = {
        "displacement_m": np.asarray(
            uh.x.array.reshape(-1, 3)[idx], dtype=np.float64,
        ),
        "w_clamp": w_nodes,
        "k_clamp_Npm3": w_nodes * float(k_clamp),
    }

    if cells.shape[0] != von_mises_Pa.shape[0]:
        raise RuntimeError(
            f"Cell count mismatch: vtk_mesh={cells.shape[0]}, "
            f"von_mises={von_mises_Pa.shape[0]}"
        )

    cell_data = {"von_mises_MPa": [von_mises_Pa / 1e6]}
    if f_vol_cell is not None:
        f_cell = np.asarray(f_vol_cell, dtype=np.float64)
        if f_cell.shape != (cells.shape[0], 3):
            raise RuntimeError(
                f"f_vol_cell shape {f_cell.shape} != "
                f"({cells.shape[0]}, 3)"
            )
        cell_data["f_vol_Npm3"] = [f_cell]

    meshio.Mesh(
        points=np.asarray(x, dtype=np.float64),
        cells=[(meshio_type, cells.astype(np.int32))],
        point_data=point_data,
        cell_data=cell_data,
    ).write(out_path)


def main():
    domain, cell_tags, facet_tags = import_mesh()
    tdim = domain.topology.dim
    print(f"cells: {domain.topology.index_map(tdim).size_global}")
    print(f"geometry nodes: {domain.geometry.x.shape[0]}")
    print(f"geometry degree: {domain.geometry.cmap.degree}")

    exterior_dofs, exterior_coords = exterior_node_indices(domain)
    print(f"exterior nodes: {exterior_coords.shape[0]}")

    data = load_mesh_sidecar()
    clamp_centers = np.asarray(data["clamp_centers"], dtype=np.float64)
    r_clamp = float(data["r_clamp"])
    eps_sigmoid = float(data["eps_sigmoid"])
    E = float(data["E"])
    nu = float(data["nu"])
    k_clamp = float(data["k_clamp"])
    rho = float(data["rho"])
    g_vec = np.asarray(data["g_vec"], dtype=np.float64)

    w_ext = clamp_weight_at(exterior_coords, clamp_centers, r_clamp, eps_sigmoid)
    print(
        f"analytic exterior w_clamp range: "
        f"[{float(w_ext.min()):.3g}, {float(w_ext.max()):.3g}]"
    )
    if "support_weight" in data.files:
        cond = data["owner_coil"] >= 0
        w_ref = data["support_weight"][cond]
        w_an = clamp_weight_at(
            data["points"][cond], clamp_centers, r_clamp, eps_sigmoid,
        )
        print(
            f"analytic vs npz support_weight max |Δ| on conductor: "
            f"{np.max(np.abs(w_ref - w_an)):.3e}"
        )

    print(f"loading {JSTRESS_PATH} for coil geometry / currents …")
    objs = load(JSTRESS_PATH)
    Jstress = objs[0] if isinstance(objs, (list, tuple)) else objs
    mesh_options = Jstress._mesh_options
    if isinstance(mesh_options, list):
        mesh_options = mesh_options[0]
    w1 = float(mesh_options["w1"])
    w2 = float(mesh_options["w2"])

    quad_pts, n_quads = physical_volume_quad_points(domain, VOL_QUAD_DEG)
    n_cells = quad_pts.shape[0]
    print(
        f"volume quads: n_cells={n_cells}, n_quads={n_quads}, "
        f"degree={VOL_QUAD_DEG}"
    )

    f_vol_q = compute_f_vol_at_quads(
        quad_pts, Jstress, rho=rho, g_vec=g_vec, w1=w1, w2=w2,
    )
    f = make_quadrature_force_function(domain, f_vol_q, VOL_QUAD_DEG)
    print(
        f"quadrature f DOFs: {f.x.array.size} "
        f"(expect {n_cells * n_quads * 3})"
    )

    print(
        f"solving: E={E:.3e} Pa, nu={nu}, k_clamp={k_clamp:.3e} N/m³, "
        f"n_clamps={clamp_centers.shape[0]}, r_clamp={r_clamp:.4g}, "
        f"eps_sigmoid={eps_sigmoid:.4g}"
    )

    time1 = time.time()
    sol = solve_elasticity_winkler(
        domain, f,
        E=E, nu=nu, k_clamp=k_clamp,
        clamp_centers=clamp_centers,
        r_clamp=r_clamp,
        eps_sigmoid=eps_sigmoid,
        vol_quad_deg=VOL_QUAD_DEG,
    )
    uh = sol["uh"]
    vm = sol["von_mises_Pa"]
    time2 = time.time()
    np.save("time_dolfinx", time2 - time1)
    print(
        f"|u|_max = "
        f"{np.linalg.norm(uh.x.array.reshape(-1, 3), axis=1).max():.3e} m"
    )
    print(
        f"von Mises range: "
        f"[{vm.min() / 1e6:.3g}, {vm.max() / 1e6:.3g}] MPa"
    )

    f_vol_cell = f_vol_q.mean(axis=1)
    write_result_vtu(
        domain, OUT_VTU,
        uh=uh, von_mises_Pa=vm,
        clamp_centers=clamp_centers,
        r_clamp=r_clamp,
        eps_sigmoid=eps_sigmoid,
        k_clamp=k_clamp,
        f_vol_cell=f_vol_cell,
    )
    print(f"wrote {OUT_VTU}")


if __name__ == "__main__":
    main()
