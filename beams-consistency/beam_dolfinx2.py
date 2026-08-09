"""Full-body dolfinx elasticity from ``full_body_fields.vtu`` + ``Jstress.json``.

Purpose-built path for the VTU+JSON workflow (no ``.msh`` / ``body_force.npz``).

The VTU carries the TET10 mesh, nodal classification fields, and FieldData for
clamp / material / gravity parameters.  ``Jstress.json`` supplies coil geometry
and currents for Lorentz body force at volume quadrature points.

Winkler clamps use analytic spheres from FieldData (same model as
``beam_dolfinx.py``), evaluated at facet quadrature points — not the nodal
``w_clamp`` point data.

DOF numbering may differ from the gmshio/``.msh`` import; compare solutions by
matching geometry coordinates.
"""

from __future__ import annotations

import time

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
from dolfinx.mesh import create_mesh

from coil_fem.magnetic import biot_savart, B_self_quadrature, lorentz_body_force

VTU_PATH = "full_body_fields.vtu"
JSTRESS_PATH = "Jstress.json"
OUT_VTU = "full_body_elasticity2.vtu"
VOL_QUAD_DEG = 4
CHUNK = 32_768
N_PHI = 4096
UV_TOL = 2e-2

_FIELD_KEYS = (
    "clamp_centers", "r_clamp", "eps_sigmoid", "k_clamp",
    "E", "nu", "rho", "g_vec",
)

_VTK_TO_MESHIO = {
    10: "tetra",
    24: "tetra10",
    71: "tetra10",
}


def load_vtu_problem(path: str = VTU_PATH):
    """Load dolfinx mesh and physics params from an enriched VTU."""
    m = meshio.read(path)
    cells = m.get_cells_type("tetra10")
    if cells.size == 0:
        raise ValueError(f"{path} has no tetra10 cells")
    missing = [k for k in _FIELD_KEYS if k not in m.field_data]
    if missing:
        raise KeyError(
            f"{path} missing FieldData {missing}. Re-run mesh export Step 4D."
        )

    el = basix.ufl.element("Lagrange", "tetrahedron", 2, shape=(3,))
    domain = create_mesh(
        MPI.COMM_WORLD,
        np.asarray(cells, dtype=np.int64),
        np.asarray(m.points, dtype=np.float64),
        ufl.Mesh(el),
    )

    fd = m.field_data
    params = {
        "clamp_centers": np.asarray(fd["clamp_centers"], dtype=np.float64).reshape(-1, 3),
        "r_clamp": float(np.asarray(fd["r_clamp"]).reshape(-1)[0]),
        "eps_sigmoid": float(np.asarray(fd["eps_sigmoid"]).reshape(-1)[0]),
        "k_clamp": float(np.asarray(fd["k_clamp"]).reshape(-1)[0]),
        "E": float(np.asarray(fd["E"]).reshape(-1)[0]),
        "nu": float(np.asarray(fd["nu"]).reshape(-1)[0]),
        "rho": float(np.asarray(fd["rho"]).reshape(-1)[0]),
        "g_vec": np.asarray(fd["g_vec"], dtype=np.float64).reshape(3),
    }
    return domain, params


def load_jstress(path: str = JSTRESS_PATH):
    """Load CoilFEMObjective and coil cross-section widths."""
    objs = load(path)
    Jstress = objs[0] if isinstance(objs, (list, tuple)) else objs
    mesh_options = Jstress._mesh_options
    if isinstance(mesh_options, list):
        mesh_options = mesh_options[0]
    return Jstress, float(mesh_options["w1"]), float(mesh_options["w2"])


def volume_quad_points(domain, vol_quad_deg: int = VOL_QUAD_DEG):
    """Physical volume quadrature coords, shape ``(n_cells, n_quads, 3)``."""
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
            f"quad packing mismatch: {arr.shape[0]} values, {n_cells} cells"
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


def _classify_points(points: np.ndarray, Jstress, *, w1: float, w2: float):
    """Conductor vs beam via (φ, u, v) pullback; returns owners, coords, Q_list."""
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
        gamma_s.append(np.asarray(fc.curve.gamma_eval(phi_s)))
        _, p, q = fc.rotated_frame_eval(phi_s)
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

            curve = Jstress.fem.meshes[i].framed_curve.curve
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


def body_force_at_quads(
    quad_pts: np.ndarray,
    Jstress,
    *,
    rho: float,
    g_vec: np.ndarray,
    w1: float,
    w2: float,
) -> np.ndarray:
    """Lorentz (+ gravity) at volume quads → shape ``(n_cells, n_quads, 3)``."""
    n_cells, n_quads, _ = quad_pts.shape
    pts = quad_pts.reshape(-1, 3)
    owner_coil, owner_sym, phi_out, u_out, v_out, Q_list = _classify_points(
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
        A = coil_mesh.cross_section_area
        I = all_currents[i]
        cross_section = {"shape": "rect", "w1": coil_mesh.w1, "w2": coil_mesh.w2}

        phi_i = phi_out[sel]
        uv_i = np.stack([u_out[sel], v_out[sel]], axis=1)
        y_i = y_base[sel]

        gd = np.asarray(fc.curve.gamma_eval(phi_i, 1))
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


def _pack_quadrature_force(domain, f_vol_q: np.ndarray, vol_quad_deg: int):
    Qe = basix.ufl.quadrature_element(
        domain.topology.cell_name(),
        value_shape=(3,),
        degree=vol_quad_deg,
    )
    f = fem.Function(fem.functionspace(domain, Qe))
    flat = np.asarray(f_vol_q, dtype=np.float64).reshape(-1)
    if flat.size != f.x.array.size:
        raise RuntimeError(
            f"f_vol packing mismatch: got {flat.size}, "
            f"quadrature space has {f.x.array.size}"
        )
    f.x.array[:] = flat
    return f


def _clamp_weight_ufl(x, centers, r_clamp: float, eps_sigmoid: float):
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


def solve_winkler(domain, f, params: dict, *, vol_quad_deg: int = VOL_QUAD_DEG):
    """Linear elasticity + analytic Winkler on the exterior; return uh, von Mises."""
    E = float(params["E"])
    nu = float(params["nu"])
    k_clamp = float(params["k_clamp"])
    clamp_centers = params["clamp_centers"]
    r_clamp = float(params["r_clamp"])
    eps_sigmoid = float(params["eps_sigmoid"])

    lam = fem.Constant(domain, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
    mu = fem.Constant(domain, E / (2.0 * (1.0 + nu)))

    degree = domain.geometry.cmap.degree
    V = fem.functionspace(domain, ("Lagrange", degree, (3,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def sigma(u_fn):
        eps = ufl.sym(ufl.grad(u_fn))
        return lam * ufl.tr(eps) * ufl.Identity(3) + 2 * mu * eps

    dx = ufl.Measure(
        "dx", domain=domain,
        metadata={"quadrature_degree": vol_quad_deg},
    )
    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * dx
    L = ufl.inner(f, v) * dx

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    ext_facets = dmesh.exterior_facet_indices(domain.topology)
    facet_tags = dmesh.meshtags(
        domain, fdim, ext_facets, np.ones(len(ext_facets), dtype=np.int32),
    )
    ds = ufl.Measure(
        "ds", domain=domain,
        subdomain_data=facet_tags, subdomain_id=1,
        metadata={"quadrature_degree": 4 if degree == 2 else 2},
    )
    x = ufl.SpatialCoordinate(domain)
    a = a + float(k_clamp) * _clamp_weight_ufl(
        x, clamp_centers, r_clamp, eps_sigmoid,
    ) * ufl.inner(u, v) * ds

    problem = LinearProblem(
        a, L, bcs=[],
        petsc_options_prefix="full_body_elasticity2",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    uh = problem.solve()

    vm_quad_degree = 2 if degree == 2 else 1
    vol_quad_el = basix.ufl.quadrature_element(
        "tetrahedron", value_shape=(), degree=vm_quad_degree,
    )

    def von_mises_ufl(u_fn):
        sig = sigma(u_fn)
        s = sig - (ufl.tr(sig) / 3.0) * ufl.Identity(3)
        return ufl.sqrt(1.5 * ufl.inner(s, s) + 1e-30)

    W = fem.functionspace(domain, vol_quad_el)
    interp_pts = W.element.interpolation_points
    if callable(interp_pts):
        interp_pts = interp_pts()
    vm_fn = fem.Function(W)
    vm_fn.interpolate(fem.Expression(von_mises_ufl(uh), interp_pts))

    n_cells = domain.topology.index_map(tdim).size_local
    n_q = vm_fn.x.array.shape[0] // n_cells
    von_mises_Pa = np.asarray(
        vm_fn.x.array.reshape(n_cells, n_q).mean(axis=1),
        dtype=np.float64,
    ).copy()
    return uh, von_mises_Pa


def _clamp_weight_at(points, centers, r_clamp, eps_sigmoid):
    from coil_fem.utils import clamp_sigmoid
    pts = jnp.asarray(points, dtype=jnp.float64)
    ctr = jnp.asarray(centers, dtype=jnp.float64)
    d_sq = jnp.sum((pts[:, None, :] - ctr[None, :, :]) ** 2, axis=-1)
    return np.asarray(
        clamp_sigmoid(d_sq, float(r_clamp), float(eps_sigmoid)).sum(axis=-1),
        dtype=np.float64,
    )


def write_result_vtu(
    domain,
    out_path: str,
    *,
    uh,
    von_mises_Pa: np.ndarray,
    params: dict,
    f_vol_cell: np.ndarray,
    atol: float = 1e-9,
) -> None:
    topology, cell_types, x = dfx_plot.vtk_mesh(domain)
    vtk_type = int(cell_types[0])
    if vtk_type not in _VTK_TO_MESHIO:
        raise ValueError(f"Unsupported VTK cell type {vtk_type}")
    meshio_type = _VTK_TO_MESHIO[vtk_type]
    n_per_cell = int(topology[0])
    cells = topology.reshape(-1, n_per_cell + 1)[:, 1:]

    degree = domain.geometry.cmap.degree
    V_s = fem.functionspace(domain, ("Lagrange", degree))
    dist, idx = cKDTree(V_s.tabulate_dof_coordinates()).query(x, k=1)
    if float(np.max(dist)) > atol:
        raise RuntimeError(
            f"VTK geometry ↔ DOF join failed: max dist "
            f"{float(np.max(dist)):.3e} m > atol {atol}"
        )

    w_nodes = _clamp_weight_at(
        np.asarray(x, dtype=np.float64),
        params["clamp_centers"],
        params["r_clamp"],
        params["eps_sigmoid"],
    )
    cell_data = {
        "von_mises_MPa": [von_mises_Pa / 1e6],
        "f_vol_Npm3": [np.asarray(f_vol_cell, dtype=np.float64)],
    }
    meshio.Mesh(
        points=np.asarray(x, dtype=np.float64),
        cells=[(meshio_type, cells.astype(np.int32))],
        point_data={
            "displacement_m": np.asarray(
                uh.x.array.reshape(-1, 3)[idx], dtype=np.float64,
            ),
            "w_clamp": w_nodes,
            "k_clamp_Npm3": w_nodes * float(params["k_clamp"]),
        },
        cell_data=cell_data,
    ).write(out_path)


def main():
    domain, params = load_vtu_problem(VTU_PATH)
    tdim = domain.topology.dim
    print(f"cells: {domain.topology.index_map(tdim).size_global}")
    print(f"geometry nodes: {domain.geometry.x.shape[0]}")
    print(f"geometry degree: {domain.geometry.cmap.degree}")

    print(f"loading {JSTRESS_PATH} …")
    Jstress, w1, w2 = load_jstress(JSTRESS_PATH)

    quad_pts, n_quads = volume_quad_points(domain, VOL_QUAD_DEG)
    n_cells = quad_pts.shape[0]
    print(
        f"volume quads: n_cells={n_cells}, n_quads={n_quads}, "
        f"degree={VOL_QUAD_DEG}"
    )

    f_vol_q = body_force_at_quads(
        quad_pts, Jstress,
        rho=params["rho"], g_vec=params["g_vec"], w1=w1, w2=w2,
    )
    f = _pack_quadrature_force(domain, f_vol_q, VOL_QUAD_DEG)

    print(
        f"solving: E={params['E']:.3e} Pa, nu={params['nu']}, "
        f"k_clamp={params['k_clamp']:.3e} N/m³, "
        f"n_clamps={params['clamp_centers'].shape[0]}, "
        f"r_clamp={params['r_clamp']:.4g}, "
        f"eps_sigmoid={params['eps_sigmoid']:.4g}"
    )
    t0 = time.time()
    uh, vm = solve_winkler(domain, f, params, vol_quad_deg=VOL_QUAD_DEG)
    print(f"solve time: {time.time() - t0:.1f} s")
    print(
        f"|u|_max = "
        f"{np.linalg.norm(uh.x.array.reshape(-1, 3), axis=1).max():.3e} m"
    )
    print(
        f"von Mises range: "
        f"[{vm.min() / 1e6:.3g}, {vm.max() / 1e6:.3g}] MPa"
    )

    write_result_vtu(
        domain, OUT_VTU,
        uh=uh, von_mises_Pa=vm, params=params,
        f_vol_cell=f_vol_q.mean(axis=1),
    )
    print(f"wrote {OUT_VTU}")


if __name__ == "__main__":
    main()
