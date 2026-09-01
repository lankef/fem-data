"""Export the solids involved in the beam-1 / coil boolean failure for ParaView.

Outputs (in ./boolean_debug/):
  beam1.stl              - beam 1 solid alone (surface mesh)
  coil{i}.stl            - each coil solid alone (two half-lofts, unfused)
  inputs_tagged.vtu      - beam1 + all coils in ONE file, cell field 'part'
                           (0 = beam1, 1..5 = coil 0..4) for transparency/coloring
  fused_coils_beam1.stl  - the FUSED union whose surface carries the sliver
  fused_coils_beam1.vtu  - same union as VTU

3D meshing of the fused union fails (that IS the bug), so all exports use the
2D surface mesh, which meshes fine and still shows the self-intersecting facet.
"""
import os
import numpy as np
import gmsh
import meshio
from simsopt import load
import coil_fem.io.gmsh as cg
from coil_fem.presets import cross_section_fns

OUT = "boolean_debug"
os.makedirs(OUT, exist_ok=True)
BAD_BEAM = 1

Jstress = load("fin_Jstress.json")[0]
coil_support = Jstress._coil_support
fem = Jstress.fem
support = fem.support
meshes = fem.meshes
sdofs = coil_support.support_dofs
curves = fem.base_curves_jax
geom = support.beam_geometry(curves, sdofs)
cs_type = getattr(coil_support, "beam_options", {}).get("cross_section_type", "solid_circle")
solid_fn = getattr(cross_section_fns, cs_type + "_solid")

x_start = np.asarray(geom["x_start"], float)
x_end = np.asarray(geom["x_end"], float)
L = np.asarray(geom["L"], float)
gamma3 = np.asarray(geom["gamma3"], float)

# which coils does beam 1 connect?
phi = np.linspace(0, 1, 2000, endpoint=False)
coil_pts = [np.asarray(m.framed_curve.curve.gamma_eval(phi)) for m in meshes]
def nearest(p):
    best = (1e9, -1)
    for ci, cp in enumerate(coil_pts):
        d = float(np.min(np.linalg.norm(cp - p, axis=1)))
        if d < best[0]:
            best = (d, ci)
    return best
ds, cs_i = nearest(x_start[BAD_BEAM])
de, ce_i = nearest(x_end[BAD_BEAM])
print(f"beam {BAD_BEAM}: start~coil{cs_i} (d={ds:.4f}), end~coil{ce_i} (d={de:.4f}), L={L[BAD_BEAM]:.4f}")

mesh_opts = fem.mesh_opts[0]
w1 = float(mesh_opts["w1"])
cs_vals = np.concatenate([np.atleast_1d(np.asarray(a, float))
                          for k in support._cross_section_dof_keys for a in sdofs[k]])
SMAX = 0.6 * 0.5 * w1
SMIN = 0.6 * 0.15 * float(cs_vals.min())

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.option.setNumber("Mesh.MeshSizeMax", SMAX)
gmsh.option.setNumber("Mesh.MeshSizeMin", SMIN)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)


def _add_beam(occ, b):
    all_b = cg._beam_solids(occ, support, sdofs, geom, solid_fn)
    drop = [dt for j, dt in enumerate(all_b) if j != b]
    if drop:
        occ.remove(drop, recursive=True)
    return all_b[b:b + 1]


def _add_coil(occ, i):
    return cg._coil_solid(occ, meshes[i])


def export_stl(builder, fname):
    gmsh.clear(); gmsh.model.add("m")
    builder(gmsh.model.occ)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)
    gmsh.write(os.path.join(OUT, fname))
    print(f"  wrote {fname}")


# individual parts as STL
export_stl(lambda occ: _add_beam(occ, BAD_BEAM), "beam1.stl")
for i in range(len(meshes)):
    export_stl(lambda occ, i=i: _add_coil(occ, i), f"coil{i}.stl")


# inputs (beam1 + all coils, UNFUSED) in one tagged VTU
def build_inputs(occ):
    parts = []  # (voltag, part_id)
    beam = _add_beam(occ, BAD_BEAM)
    occ.synchronize()
    for (_, t) in beam:
        parts.append((t, 0))
    for i in range(len(meshes)):
        cs = _add_coil(occ, i)
        occ.synchronize()
        for (_, t) in cs:
            parts.append((t, i + 1))
    return parts

gmsh.clear(); gmsh.model.add("inputs")
occ = gmsh.model.occ
parts = build_inputs(occ)
occ.synchronize()
# tag boundary surfaces of each solid by part id
for voltag, pid in parts:
    surfs = [t for (_, t) in gmsh.model.getBoundary([(3, voltag)], oriented=False)]
    if surfs:
        gmsh.model.addPhysicalGroup(2, surfs, tag=1000 * (pid + 1) + voltag)
gmsh.model.mesh.generate(2)
tmp_msh = os.path.join(OUT, "_inputs.msh")
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.write(tmp_msh)
m = meshio.read(tmp_msh)
# physical tag -> part id
if "gmsh:physical" in m.cell_data_dict:
    for ck, arr in m.cell_data_dict["gmsh:physical"].items():
        pass
tri_blocks = [(ct, cd) for ct, cd in zip([c.type for c in m.cells], m.cells)]
part_field = []
phys = m.cell_data.get("gmsh:physical")
if phys is not None:
    for blk in phys:
        part_field.append((blk // 1000 - 1).astype(np.int32))
    m.cell_data["part"] = part_field
meshio.write(os.path.join(OUT, "inputs_tagged.vtu"), m)
os.remove(tmp_msh)
print("  wrote inputs_tagged.vtu")


# fused union (the failing geometry) -> surface mesh STL + VTU
gmsh.clear(); gmsh.model.add("fused")
occ = gmsh.model.occ
coils = []
for i in range(len(meshes)):
    coils += _add_coil(occ, i)
beam = _add_beam(occ, BAD_BEAM)
solids = coils + beam
occ.fuse(solids[:1], solids[1:])
occ.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write(os.path.join(OUT, "fused_coils_beam1.stl"))
gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
tmp2 = os.path.join(OUT, "_fused.msh")
gmsh.write(tmp2)
meshio.write(os.path.join(OUT, "fused_coils_beam1.vtu"), meshio.read(tmp2))
os.remove(tmp2)
print("  wrote fused_coils_beam1.stl / .vtu")

gmsh.finalize()
print("done ->", os.path.abspath(OUT))
