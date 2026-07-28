"""Beams-optimization glue module.

Rather than re-implementing (and duplicating) the penalty-assembly /
Fourier-continuation machinery, this module dynamically imports
``fixed-continuation/optimization.py`` and wraps its ``run_filament_free`` /
``run_continuation`` with the CoilFEMObjective parameters (mesh/material/
gravity/problem/physics options + the ``CoilSupportBeams`` beam-network
kwargs) recorded in ``beams-consistency/Jstress.json``.
"""

import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent

# ── Read Jstress.json's stored constructor parameters ───────────────────────
# Simsopt's JSON serializer stores every constructor argument as plain data
# under ``simsopt_objs``, so we can recover the option dicts with a plain
# ``json.load`` -- no need to instantiate the (GPU-heavy) FEM objects just to
# read their settings.
_JSTRESS_PATH = _REPO_ROOT / 'beams-consistency' / 'Jstress.json'
with open(_JSTRESS_PATH) as _f:
    _jstress_objs = json.load(_f)['simsopt_objs']

_fem_obj_spec = _jstress_objs['CoilFEMObjective1']
_support_spec = _jstress_objs['CoilSupportBeams1']

# Mesh / material / gravity / Winkler-BC / solver / physics options forwarded
# to CoilFEMObjective -- identical to Jstress.json (gravity is zero there).
mesh_options     = _fem_obj_spec['mesh_options']
material_options = _fem_obj_spec['material_options']
gravity_options  = _fem_obj_spec['gravity_options']
problem_options  = _fem_obj_spec['problem_options']
physics_options  = _fem_obj_spec['physics_options']
coupling         = _fem_obj_spec['coupling']

# CoilSupportBeams kwargs reproducing the exact beam network + fixed clamps
# used in Jstress.json (n_beam_cc=4, n_beam_cf=0, r_beam=0.06 fixed,
# thetas_orientation_cc fixed, fixed clamps enabled).
beam_support_kwargs = dict(
    beam_options        = _support_spec['beam_options'],
    fixed_clamp_options = _support_spec['fixed_clamp_options'],
    fixed_dof_names     = tuple(_support_spec['fixed_dof_names']),
    r_beam              = _support_spec['r_beam'],
)

# ── Import fixed-continuation/optimization.py's reusable machinery ──────────
# Loaded under a distinct module name (instead of a plain ``import
# optimization``) so it can't collide with *this* module -- the two
# beams-optimization scripts import this file itself as "optimization".
# Executing it runs its own module-level setup once (load_eq, get_data,
# target constants), exactly as when fixed-continuation's own scripts import
# it directly; it resolves its own wout.nc relative to its own file, so this
# works regardless of the caller's current working directory.
_FC_PATH = _REPO_ROOT / 'fixed-continuation' / 'optimization.py'
_fc_spec = importlib.util.spec_from_file_location(
    '_fixed_continuation_optimization', _FC_PATH,
)
_fc = importlib.util.module_from_spec(_fc_spec)
sys.modules[_fc_spec.name] = _fc
_fc_spec.loader.exec_module(_fc)

# Match beams-consistency/Jstress.json's resolution (order=10, ppp=16 -> 160
# quadpoints/coil) instead of fixed-continuation's own default of 80.
# ``ppp_for_target_quadpoints``/``increase_base_curve_order`` in the imported
# module look this global up dynamically (not captured at import time), so
# overriding it here takes effect immediately for every call below.
_fc.TARGET_QUADPOINTS_PER_COIL = 160

# Re-exported unchanged.
load_eq = _fc.load_eq
ppp_for_target_quadpoints = _fc.ppp_for_target_quadpoints
increase_base_curve_order = _fc.increase_base_curve_order
coil_per_half_fp = _fc.coil_per_half_fp


def run_filament_free(*args, **kwargs):
    """``fixed-continuation``'s ``run_filament_free``, defaulting its FEM
    options to the beams (``CoilSupportBeams``) values read from
    ``Jstress.json`` above."""
    kwargs.setdefault('mesh_options', mesh_options)
    kwargs.setdefault('material_options', material_options)
    kwargs.setdefault('gravity_options', gravity_options)
    kwargs.setdefault('problem_options', problem_options)
    kwargs.setdefault('physics_options', physics_options)
    kwargs.setdefault('coupling', coupling)
    return _fc.run_filament_free(*args, **kwargs)


def run_continuation(*args, **kwargs):
    """``fixed-continuation``'s ``run_continuation``, defaulting its FEM
    options to the beams (``CoilSupportBeams``) values read from
    ``Jstress.json`` above."""
    kwargs.setdefault('mesh_options', mesh_options)
    kwargs.setdefault('material_options', material_options)
    kwargs.setdefault('gravity_options', gravity_options)
    kwargs.setdefault('problem_options', problem_options)
    kwargs.setdefault('physics_options', physics_options)
    kwargs.setdefault('coupling', coupling)
    return _fc.run_continuation(*args, **kwargs)
