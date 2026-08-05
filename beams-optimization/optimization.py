"""Beams-optimization glue module.

Rather than re-implementing (and duplicating) the penalty-assembly /
Fourier-continuation machinery, this module dynamically imports
``fixed-continuation/optimization.py`` and wraps its ``run_filament_free`` /
``run_continuation`` with the CoilFEMObjective parameters (mesh/material/
gravity/problem/physics options + the ``CoilSupportBeams`` beam-network
kwargs) from ``beam-options.json``.

Shared helpers (``load_eq``, …) come from ``fem-data/opt_utils.py``.
"""

import importlib.util
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import opt_utils
from opt_utils import load_eq

# Match beams-consistency resolution (~110 quadpoints/coil) instead of the
# shared default of 80. Looked up dynamically by ppp_for_target_quadpoints /
# increase_base_curve_order inside fixed-continuation's run_continuation.
opt_utils.TARGET_QUADPOINTS_PER_COIL = 110

_OPTIONS_PATH = _REPO_ROOT / 'beam-options.json'
opts = json.load(open(_OPTIONS_PATH))

mesh_options = opts['mesh_options']
material_options = opts['material_options']
gravity_options = opts['gravity_options']
problem_options = dict(opts['problem_options'])
physics_options = opts['physics_options']
coupling = opts['coupling']

beam_support_kwargs = dict(
    beam_options=dict(opts['beam_options']),
    fixed_clamp_options=dict(opts['fixed_clamp_options']),
    fixed_dof_names=tuple(opts['fixed_dof_names']),
    r_beam=opts['r_beam'],
)

# ── Import fixed-continuation/optimization.py's reusable machinery ──────────
# Loaded under a distinct module name (instead of a plain ``import
# optimization``) so it can't collide with *this* module -- the two
# beams-optimization scripts import this file itself as "optimization".
_FC_PATH = _REPO_ROOT / 'fixed-continuation' / 'optimization.py'
_fc_spec = importlib.util.spec_from_file_location(
    '_fixed_continuation_optimization', _FC_PATH,
)
_fc = importlib.util.module_from_spec(_fc_spec)
sys.modules[_fc_spec.name] = _fc
_fc_spec.loader.exec_module(_fc)

coil_per_half_fp = _fc.coil_per_half_fp


def run_filament_free(*args, **kwargs):
    """``fixed-continuation``'s ``run_filament_free``, defaulting its FEM
    options to the beams values from ``beam-options.json``."""
    kwargs.setdefault('mesh_options', mesh_options)
    kwargs.setdefault('material_options', material_options)
    kwargs.setdefault('gravity_options', gravity_options)
    kwargs.setdefault('problem_options', problem_options)
    kwargs.setdefault('physics_options', physics_options)
    kwargs.setdefault('coupling', coupling)
    return _fc.run_filament_free(*args, **kwargs)


def run_continuation(*args, **kwargs):
    """``fixed-continuation``'s ``run_continuation``, defaulting its FEM
    options to the beams values from ``beam-options.json``."""
    kwargs.setdefault('mesh_options', mesh_options)
    kwargs.setdefault('material_options', material_options)
    kwargs.setdefault('gravity_options', gravity_options)
    kwargs.setdefault('problem_options', problem_options)
    kwargs.setdefault('physics_options', physics_options)
    kwargs.setdefault('coupling', coupling)
    return _fc.run_continuation(*args, **kwargs)
