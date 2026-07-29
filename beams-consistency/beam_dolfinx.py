"""Import the fused full-body mesh, load body force, and run dolfinx elasticity.

Reads the fused, symmetry-expanded coil+beam mesh (``full_mesh.msh``, TET10)
into dolfinx, joins nodal fields from ``body_force.npz`` onto a P2 vector
function space by KD-tree coordinate matching, solves linear elasticity with
a Winkler spring BC, and writes ``full_body_elasticity.vtu``.

Winkler weights are evaluated from analytic clamp spheres
(``clamp_centers`` / ``r_clamp`` / ``eps_sigmoid`` in the npz) at facet
quadrature points — the same ``clamp_sigmoid`` used by CoilFEM — rather than
by interpolating a nodal P2 weight field (which overshoots at the patch rim).

No thermal eigenstrain is applied here (``body_force.npz`` carries no
``itc``); CoilFEM's notebook case includes ``itc = 0.0029`` prestress.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import ufl
import basix.ufl
import meshio
import time
from mpi4py import MPI
from scipy.spatial import cKDTree
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx import plot as dfx_plot
from dolfinx.fem.petsc import LinearProblem
try:
    from dolfinx.io import gmshio  # dolfinx <= 0.9
except ImportError:  # dolfinx >= 0.10 renamed gmshio -> gmsh
    from dolfinx.io import gmsh as gmshio

from coil_fem.utils import clamp_sigmoid

MESH_PATH = "full_mesh.msh"
BODY_FORCE_PATH = "body_force.npz"
OUT_VTU = "full_body_elasticity.vtu"

# VTK cell-type integers returned by dolfinx.plot.vtk_mesh → meshio names.
_VTK_TO_MESHIO = {
    10: "tetra",                 # VTK_TETRA
    24: "tetra10",               # VTK_QUADRATIC_TETRA
    71: "tetra10",               # VTK_LAGRANGE_TETRAHEDRON (order 2 → 10 nodes)
}


def import_mesh(path: str = MESH_PATH):
    """Read the fused device mesh into a dolfinx ``Mesh``.

    Parameters
    ----------
    path : str
        Path to the gmsh ``.msh`` file written by ``full_body.ipynb`` Step 3.

    Returns
    -------
    domain : dolfinx.mesh.Mesh
    cell_tags, facet_tags : dolfinx.mesh.MeshTags
        Physical-group tags carried over from gmsh (only the ``"device"``
        volume group is set in the notebook).
    """
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
    """Return the DOF indices and coordinates of every exterior mesh node.

    Uses a scalar Lagrange space at the same degree as the mesh geometry
    (``domain.geometry.cmap.degree``) so its DOF coordinates coincide with
    every geometry node — including TET10 midside nodes — not just the
    corner vertices that ``exterior_facet_indices`` alone identifies.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh

    Returns
    -------
    exterior_dofs : numpy.ndarray, shape (n_exterior,)
    exterior_coords : numpy.ndarray, shape (n_exterior, 3)
    """
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
    """Sum of ``clamp_sigmoid`` over clamp centres at each query point.

    Matches :func:`coil_fem.simsopt.optimizables._fixed_weights` /
    ``Support.compute_weights`` for sphere clamps.

    Parameters
    ----------
    points : numpy.ndarray, shape (N, 3)
    centers : numpy.ndarray, shape (n_clamps, 3)
    r_clamp : float
    eps_sigmoid : float

    Returns
    -------
    numpy.ndarray, shape (N,)
    """
    pts = jnp.asarray(points, dtype=jnp.float64)
    ctr = jnp.asarray(centers, dtype=jnp.float64)
    d_sq = jnp.sum((pts[:, None, :] - ctr[None, :, :]) ** 2, axis=-1)
    return np.asarray(
        clamp_sigmoid(d_sq, float(r_clamp), float(eps_sigmoid)).sum(axis=-1),
        dtype=np.float64,
    )


def clamp_weight_ufl(x, centers, r_clamp: float, eps_sigmoid: float):
    """UFL weight matching :func:`clamp_weight_at` at facet quadrature points.

    Built from ``SpatialCoordinate`` so FFCx evaluates the analytic sigmoid at
    each facet quadrature point (no P2 interpolation of a nodal weight field).
    """
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


def load_body_force(domain, path: str = BODY_FORCE_PATH, *, atol: float = 1e-9):
    """Load nodal body force from ``body_force.npz`` onto a P2 vector Function.

    Joins by KD-tree nearest-neighbour on coordinates because dolfinx
    renumbers / repartitions nodes on import.  P2 dofs coincide with TET10
    geometry nodes, so the match should be exact to roundoff.

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    path : str
        Path to the ``.npz`` written by ``full_body.ipynb`` Step 4C.
    atol : float
        Max allowed KD-tree distance (m); raises if exceeded.

    Returns
    -------
    f : dolfinx.fem.Function
        Vector P2 function with body-force density [N/m³].
    data : np.lib.npyio.NpzFile
        Full archive (``clamp_centers``, ``owner_coil``, ``B_self``, …).
    """
    data = np.load(path)
    if "clamp_centers" not in data.files:
        raise KeyError(
            f"{path} is missing 'clamp_centers' (and likely 'r_clamp' / "
            f"'eps_sigmoid'). Re-run the mesh export (mesh.ipynb Step 4) "
            f"or patch the npz with physical-frame clamp centres."
        )
    degree = domain.geometry.cmap.degree
    V = fem.functionspace(domain, ("Lagrange", degree, (domain.geometry.dim,)))
    V_s = fem.functionspace(domain, ("Lagrange", degree))

    # Match on the scalar-space coordinates (vector spaces repeat coords
    # per component in blocked layout).
    coords_s = V_s.tabulate_dof_coordinates()
    dist, idx = cKDTree(data["points"]).query(coords_s, k=1)
    if float(np.max(dist)) > atol:
        raise RuntimeError(
            f"body_force.npz coordinate join failed: max dist "
            f"{float(np.max(dist)):.3e} m > atol {atol}"
        )

    f = fem.Function(V)
    f.x.array.reshape(-1, domain.geometry.dim)[:] = np.asarray(
        data["f_vol"][idx], dtype=np.float64,
    )

    return f, data


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
):
    """Solve linear elasticity with analytic clamp-sphere Winkler BC.

    Body-force volume quadrature is left to UFL/FFCx auto-estimation
    (``f`` and ``v`` are both P2).  The Winkler face quadrature is forced to
    degree 4 on TET10 so the local face mass matrix on TRI6 has full rank.
    Weights use the analytic clamp sigmoid at facet quadrature points (same
    formula as :func:`coil_fem.utils.clamp_sigmoid`).

    No thermal eigenstrain is applied (``body_force.npz`` has no ``itc``).

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    f : dolfinx.fem.Function
        Vector P2 body-force density [N/m³] on ``domain``.
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    k_clamp : float
        Foundation modulus [N/m³]; effective spring is ``k_clamp * w(x)``.
    clamp_centers : numpy.ndarray, shape (n_clamps, 3)
        Clamp sphere centres in the mesh frame.
    r_clamp : float
        Clamp sphere radius [m].
    eps_sigmoid : float
        Relative sigmoid width (see :func:`coil_fem.utils.clamp_sigmoid`).

    Returns
    -------
    dict
        ``uh`` : dolfinx.fem.Function — nodal displacement [m].
        ``von_mises_Pa`` : numpy.ndarray, shape ``(n_cells,)`` — cell-mean
        von Mises stress [Pa] in dolfinx cell order.
    """
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
        # Same constitutive law used for von Mises (no thermal eigenstrain).
        return sigma_elastic(u_fn)

    a = ufl.inner(sigma_elastic(u), ufl.sym(ufl.grad(v))) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Winkler spring on every exterior facet; w(x) analytic at facet quads.
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    ext_facets = dmesh.exterior_facet_indices(domain.topology)
    facet_tags = dmesh.meshtags(
        domain, fdim, ext_facets, np.ones(len(ext_facets), dtype=np.int32),
    )
    # Match CoilFEM / shared.py: TET10 TRI6 faces need ≥6 quad points for a
    # full-rank local Winkler mass matrix (gauss_order=4 → degree 4).
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

    # Von Mises at volume Gauss points, then cell mean (shared.py pattern;
    # no coil-fem cell permutation — stay in dolfinx native cell order).
    vol_quad_degree = 2 if degree == 2 else 1
    try:
        vol_quad_el = basix.ufl.quadrature_element(
            "tetrahedron", value_shape=(), degree=vol_quad_degree,
        )
    except TypeError:
        vol_quad_el = basix.ufl.quadrature_element(
            "tetrahedron", "default", vol_quad_degree, (),
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
    f=None,
    B_self: np.ndarray | None = None,
    B_ext: np.ndarray | None = None,
    npz_points: np.ndarray | None = None,
    atol: float = 1e-9,
) -> None:
    """Write a ``save_run_vtu``-style VTU for the fused-body dolfinx solve.

    Connectivity and geometry come from ``dolfinx.plot.vtk_mesh`` (VTK-ready,
    domain's own order).  Point fields are KD-matched from dolfinx DOF
    coordinates onto the VTK geometry nodes.  ``w_clamp`` is the analytic
    clamp sigmoid at each VTK node (not a P2-interpolated FE field).

    Parameters
    ----------
    domain : dolfinx.mesh.Mesh
    out_path : str
    uh : dolfinx.fem.Function
        Displacement solution.
    von_mises_Pa : numpy.ndarray, shape (n_cells,)
        Cell-mean von Mises [Pa] in dolfinx cell order.
    clamp_centers : numpy.ndarray, shape (n_clamps, 3)
    r_clamp, eps_sigmoid : float
    k_clamp : float
        Foundation modulus [N/m³].
    f : dolfinx.fem.Function or None
        Body-force density; written as ``f_vol_Npm3`` when provided.
    B_self, B_ext : numpy.ndarray or None, shape (N_npz, 3)
        Magnetic fields from ``body_force.npz``; joined via ``npz_points``.
    npz_points : numpy.ndarray or None, shape (N_npz, 3)
        Coordinates accompanying ``B_self`` / ``B_ext``.
    atol : float
        Max allowed KD-tree distance (m).
    """
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
    if f is not None:
        point_data["f_vol_Npm3"] = np.asarray(
            f.x.array.reshape(-1, 3)[idx], dtype=np.float64,
        )
    if B_self is not None:
        if npz_points is None:
            raise ValueError("npz_points is required when B_self is provided")
        dist2, idx2 = cKDTree(npz_points).query(x, k=1)
        if float(np.max(dist2)) > atol:
            raise RuntimeError(
                f"B-field npz coordinate join failed: max dist "
                f"{float(np.max(dist2)):.3e} m > atol {atol}"
            )
        point_data["B_self_T"] = np.asarray(B_self[idx2], dtype=np.float64)
        if B_ext is not None:
            point_data["B_ext_T"] = np.asarray(B_ext[idx2], dtype=np.float64)

    if cells.shape[0] != von_mises_Pa.shape[0]:
        raise RuntimeError(
            f"Cell count mismatch: vtk_mesh={cells.shape[0]}, "
            f"von_mises={von_mises_Pa.shape[0]}"
        )

    meshio.Mesh(
        points=np.asarray(x, dtype=np.float64),
        cells=[(meshio_type, cells.astype(np.int32))],
        point_data=point_data,
        cell_data={"von_mises_MPa": [von_mises_Pa / 1e6]},
    ).write(out_path)


def main():
    domain, cell_tags, facet_tags = import_mesh()
    tdim = domain.topology.dim
    print(f"cells: {domain.topology.index_map(tdim).size_global}")
    print(f"geometry nodes: {domain.geometry.x.shape[0]}")
    print(f"geometry degree: {domain.geometry.cmap.degree}")

    exterior_dofs, exterior_coords = exterior_node_indices(domain)
    print(f"exterior nodes: {exterior_coords.shape[0]}")

    f, data = load_body_force(domain)
    clamp_centers = np.asarray(data["clamp_centers"], dtype=np.float64)
    r_clamp = float(data["r_clamp"])
    eps_sigmoid = float(data["eps_sigmoid"])

    print(
        f"|f|_max on mesh: "
        f"{np.linalg.norm(f.x.array.reshape(-1, 3), axis=1).max():.3e} N/m³"
    )
    print(
        f"conductor fraction (from npz): "
        f"{100 * float(np.mean(data['owner_coil'] >= 0)):.1f}%"
    )
    w_ext = clamp_weight_at(exterior_coords, clamp_centers, r_clamp, eps_sigmoid)
    print(
        f"analytic exterior w_clamp range: "
        f"[{float(w_ext.min()):.3g}, {float(w_ext.max()):.3g}]"
    )
    if "support_weight" in data.files:
        # Sanity: nodal npz weights vs analytic at the same npz coordinates.
        cond = data["owner_coil"] >= 0
        w_ref = data["support_weight"][cond]
        w_an = clamp_weight_at(
            data["points"][cond], clamp_centers, r_clamp, eps_sigmoid,
        )
        err = np.max(np.abs(w_ref - w_an))
        print(f"analytic vs npz support_weight max |Δ| on conductor: {err:.3e}")

    E = float(data["E"])
    nu = float(data["nu"])
    k_clamp = float(data["k_clamp"])
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

    write_result_vtu(
        domain, OUT_VTU,
        uh=uh, von_mises_Pa=vm,
        clamp_centers=clamp_centers,
        r_clamp=r_clamp,
        eps_sigmoid=eps_sigmoid,
        k_clamp=k_clamp,
        f=f,
        B_self=data["B_self"], B_ext=data["B_ext"],
        npz_points=data["points"],
    )
    print(f"wrote {OUT_VTU}")


if __name__ == "__main__":
    main()
