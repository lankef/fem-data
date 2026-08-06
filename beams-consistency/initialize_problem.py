#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gmsh
from simsopt.configs import get_data
from simsopt.mhd import Vmec
from simsopt import save
import json
import numpy as np
import jax
import time
from pathlib import Path

# Loading the W7-X standard configuration 
# plasma surface. wout file comes from Landreman's
# VMEC equilibrium archive: 
# https://github.com/landreman/vmec_equilibria/blob/master/W7-X/Standard/
eq = Vmec('../fixed-continuation/wout.nc', keep_all_files=True)
# Adjusting resolution and taking a half-field-period
n_phi = 25
n_theta = 50
plasma_surface = type(eq.boundary)(
    nfp=eq.boundary.nfp, stellsym=eq.boundary.stellsym,
    mpol=eq.boundary.mpol, ntor=eq.boundary.ntor,
    quadpoints_phi=np.linspace(0, 1/2/eq.boundary.nfp, n_phi, endpoint=False),
    quadpoints_theta=np.linspace(0, 1, n_theta, endpoint=False),
)
plasma_surface.set_dofs(eq.boundary.get_dofs())

# Loading the W7-X coils.
coil_per_half_fp = 5
# The number of quadpoints here controls the mesh
# density in coil-fem. The meshing routine will
# try choose the cell number in the cross section 
# to make the aspect ratio close to one. Please see 
# the next sections for details. The resolution is
# low here to fit on a 8GB RTX 4060.  
curves, currents, axis, nfp, bs = get_data(
    'w7x', coil_order=10, points_per_period=16
)
base_curves = curves[:coil_per_half_fp]
base_currents = currents[:coil_per_half_fp]


# ## Defining `coil-fem` penalty

# ### FEM options and material properties

# In[2]:


_OPTIONS_PATH = Path(__file__).resolve().parent.parent / 'beam-options.json'
opts = json.load(open(_OPTIONS_PATH))
mesh_options = opts['mesh_options']
material_options = {**opts['material_options'], 'itc': 0.0029}
gravity_options = opts['gravity_options']
problem_options = opts['problem_options']
physics_options = opts['physics_options']
beam_options = opts['beam_options']
# Consistency case uses a larger clamp radius than the beams preset.
fixed_clamp_options = {
    **opts['fixed_clamp_options'],
    'r_clamp': 1.73 * mesh_options['w1'],
}
mesh_scale = 0.5


# ### Support structures
# Before beginning optimization, we first define a support structure for the
# coilset. Support clamps in `coil-fem` are simsopt `Optimizables`. In this
# example, we use `CoilSupportBeams`. In addition to fixed clamps, 
# this optimizable supports two types of beams: 
# 1. Coil-coil beams that connects two adjacent coils
# 2. Coil-foundation beams that connects a coil to a fixed point in space.

# In[3]:


# In[4]:


from simsopt.field import Coil
from coil_fem.simsopt import CoilSupportBeams

# One support object covers the whole base coilset
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


# ### Defining the simsopt objective
# We now defined the simsopt objective for `coil-fem`. The `CoilFEMObjective`
# simsopt `Optimizable` handles all forward solves and gradient calculations.
# Each `CoilFEMObjective` maintains its own FEM problem. 
# 
# The mesh density is controlled by the centerlines `quadpoints` counts. 
# the number of cells in the cross section is automatically calculated to 
# produce mesh cells so that each cell's aspect ratio is roughly
# `mesh_options['aspect_ratio']`.
# 
# **Note:** When multiple `coil-fem` penalties are needed, please initialize one
# `CoilFEMObjective` instance with a list of `metrics` and `metric_weights` 
# instead of defining multiple `CoilFEMObjective` instances. This way, `coil-fem`
# can compute all metrics and gradients in one FEM pass. Creating multiple 
# `CoilFEMObjective` instances for the same problem will cause `coil-fem` to repeat 
# the same FEM computation. This will cause unnecessary memory and time usage. 
# 

# In[5]:


from coil_fem.simsopt import CoilFEMObjective

# The Simsopt wrapper for a differentiable FEM problem.
# It behaves like a simsopt objective.
Jstress = CoilFEMObjective(
    coil_support,
    metrics          = ('l2_von_mises',),
    metric_weights   = (1.,),
    mesh_options     = mesh_options,
    material_options = material_options,
    gravity_options  = gravity_options,
    problem_options  = problem_options,
    physics_options  = physics_options,
    coupling         = opts['coupling'],
)
save([Jstress], 'Jstress.json')
print('# mesh node for all coils:', Jstress.n_nodes)
print('# mesh cell for all coils:', Jstress.n_cells)
# Viusualizing the mesh nodes and support clamps before optimization
Jstress.save_support_vtu('supports')
