"""Shared dump/wrap helpers for copy-only crash instrumentation.

Used by ``audit_init_constraints.py`` and ``optimize_beams_debug.py``.
Does not change the production ``optimize_beams.py`` path.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import qr as scipy_qr
from scipy.optimize import NonlinearConstraint


LOGS_DIR = Path("logs")

DEBUG_FLAG_NAMES = (
    "DEBUG_DUMMY_OBJ",
    "DEBUG_SKIP_FEM",
    "DEBUG_NO_NL_CONS",
    "DEBUG_DROP_JBCD",
    "DEBUG_FIX_THETAS",
    "DEBUG_INTERIOR_THETA",
)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "True", "yes", "YES"}


def print_debug_flags(names=DEBUG_FLAG_NAMES):
    print("Debug flags:")
    for name in names:
        print(f"  {name}={int(env_flag(name))}")


def _to_jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        if np.isnan(x):
            return "nan"
        if np.isinf(x):
            return "inf" if x > 0.0 else "-inf"
        return x
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return str(value)


def finite_stats(arr) -> dict:
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0:
        return {"n_nan": 0, "n_inf": 0, "absmax": 0.0, "size": 0}
    return {
        "n_nan": int(np.isnan(a).sum()),
        "n_inf": int(np.isinf(a).sum()),
        "absmax": _to_jsonable(np.nanmax(np.abs(a))),
        "size": int(a.size),
    }


def local_dof_name(name: str) -> str:
    return name.split(":", 1)[-1]


def is_theta_cc_name(name: str) -> bool:
    return local_dof_name(name).split("(", 1)[0] == "thetas_orientation_cc"


class DebugSession:
    """Eval counters and last-x for crash dumps."""

    def __init__(self, logs_dir=LOGS_DIR):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.counts = defaultdict(int)
        self.last_x = None
        self.last_tag = None

    def record(self, tag: str, x):
        self.counts[tag] += 1
        self.last_x = np.asarray(x, dtype=float).copy()
        self.last_tag = tag

    def snapshot(self) -> dict:
        return {
            "counts": dict(self.counts),
            "last_tag": self.last_tag,
            "last_x": None if self.last_x is None else self.last_x.tolist(),
        }


def write_preflight(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def write_crash_state(path, exc_type, exc, tb, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exc_type": getattr(exc_type, "__name__", str(exc_type)),
        "exc": str(exc),
        "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
    print(f"Wrote {path}", file=sys.stderr)


def install_excepthook(get_state, logs_dir=LOGS_DIR):
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    orig = sys.excepthook

    def hook(exc_type, exc, tb):
        try:
            extra = {}
            if get_state is not None:
                extra = get_state() or {}
            write_crash_state(logs_dir / "crash_state.json", exc_type, exc, tb, extra)
        except Exception as dump_err:  # noqa: BLE001 — never mask the original
            print(f"crash dump failed: {dump_err}", file=sys.stderr)
        orig(exc_type, exc, tb)

    sys.excepthook = hook


def constraint_from_optimizable_logged(obj, lb, ub, tag, session: DebugSession):
    """``NonlinearConstraint`` that dumps the first non-finite ``dJ``."""

    def fun(x):
        session.record(f"{tag}_fun", x)
        obj.x = x
        return obj.J()

    def jac(x):
        session.record(f"{tag}_jac", x)
        obj.x = x
        dJ = np.asarray(obj.dJ(), dtype=float)
        if not np.isfinite(dJ).all():
            out = session.logs_dir / f"bad_{tag}_jac.npz"
            np.savez(out, x=np.asarray(x, dtype=float), dJ=dJ)
            print(f"Wrote {out}", file=sys.stderr)
            stats = finite_stats(dJ)
            raise ValueError(
                f"nonfinite {tag}.dJ n_nan={stats['n_nan']} n_inf={stats['n_inf']}"
            )
        return dJ

    return NonlinearConstraint(fun, lb, ub, jac=jac)


def logged_fun(fun, session: DebugSession, tag: str = "fun"):
    """Wrap ``fun(x) -> (J, grad)`` and dump the first non-finite eval."""

    def wrapped(x):
        session.record(tag, x)
        J, grad = fun(x)
        J = float(np.asarray(J, dtype=float))
        grad = np.asarray(grad, dtype=float)
        if not (np.isfinite(J) and np.isfinite(grad).all()):
            out = session.logs_dir / "bad_fun.npz"
            np.savez(out, x=np.asarray(x, dtype=float), J=np.asarray(J), g=grad)
            print(f"Wrote {out}", file=sys.stderr)
            stats = finite_stats(grad)
            raise ValueError(
                f"nonfinite {tag} J={J} n_nan={stats['n_nan']} n_inf={stats['n_inf']}"
            )
        return J, grad

    return wrapped


def hinge_stats(name, obj) -> dict:
    J = float(np.asarray(obj.J(), dtype=float))
    dJ = np.asarray(obj.dJ(), dtype=float)
    return {
        "name": name,
        "J": _to_jsonable(J),
        "len_dJ": int(dJ.size),
        **finite_stats(dJ),
    }


def jbcd_geometry_stats(Jbcd) -> dict:
    cdofs, sdofs = Jbcd._read_dofs()
    curves_jax = Jbcd._curves_jax(cdofs)
    geom = Jbcd._support.beam_geometry(curves_jax, sdofs)
    L = np.asarray(geom["L"], dtype=float)
    _x_a, _x_b, active = Jbcd._effective_segments(geom)
    active = np.asarray(active)
    return {
        "dead_length": float(Jbcd.dead_length),
        "minimum_distance": float(Jbcd.minimum_distance),
        "min_L": _to_jsonable(float(np.min(L))) if L.size else None,
        "max_L": _to_jsonable(float(np.max(L))) if L.size else None,
        "n_beams": int(L.size),
        "n_active_spans": int(np.sum(active)),
        "L": L.tolist(),
        "active": np.asarray(active, dtype=bool).tolist(),
    }


def bound_audit(dofs, lb, ub, dof_names) -> dict:
    dofs = np.asarray(dofs, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    names = list(dof_names)
    on_lb, on_ub, violated = [], [], []
    for i, name in enumerate(names):
        x, lo, hi = dofs[i], lb[i], ub[i]
        if np.isfinite(lo) and x < lo - 1e-14:
            violated.append({"name": name, "x": float(x), "bound": float(lo), "side": "lb"})
        if np.isfinite(hi) and x > hi + 1e-14:
            violated.append({"name": name, "x": float(x), "bound": float(hi), "side": "ub"})
        if np.isfinite(lo) and abs(x - lo) <= 1e-15:
            on_lb.append(name)
        if np.isfinite(hi) and abs(x - hi) <= 1e-15:
            on_ub.append(name)
    return {
        "n_on_lb": len(on_lb),
        "n_on_ub": len(on_ub),
        "n_violated": len(violated),
        "n_thetas_on_lb": sum(1 for n in on_lb if is_theta_cc_name(n)),
        "on_lb": on_lb,
        "on_ub": on_ub,
        "violated": violated,
    }


def linear_sum_audit(linear_cons, dofs) -> dict:
    A = np.asarray(linear_cons.A, dtype=float)
    dofs = np.asarray(dofs, dtype=float)
    if A.size == 0:
        return {"n_rows": 0, "sums": [], "n_violated": 0}
    sums = A @ dofs
    ub = np.asarray(linear_cons.ub, dtype=float)
    viol = sums > ub + 1e-14
    return {
        "n_rows": int(A.shape[0]),
        "sums": sums.tolist(),
        "ub": ub.tolist(),
        "n_violated": int(np.sum(viol)),
    }


def try_qr(A) -> dict:
    A = np.asarray(A, dtype=float)
    info = {"shape": list(A.shape), **finite_stats(A)}
    try:
        scipy_qr(A.T, pivoting=True, mode="economic")
        info["ok"] = True
    except Exception as exc:  # noqa: BLE001 — record whatever SciPy raises
        info["ok"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def stacked_constraint_qr(linear_cons, hinge_dJs) -> dict:
    rows = [np.asarray(linear_cons.A, dtype=float)]
    for dJ in hinge_dJs:
        rows.append(np.atleast_2d(np.asarray(dJ, dtype=float)))
    return try_qr(np.vstack(rows))


def apply_interior_theta(opt, shift=0.05):
    """Nudge free ``thetas_orientation_cc`` off ``lb`` (clipped to ``ub``)."""
    names = list(opt.dof_names)
    x = np.asarray(opt.x, dtype=float).copy()
    _lb, ub = opt.bounds
    ub = np.asarray(ub, dtype=float)
    n_moved = 0
    for i, name in enumerate(names):
        if is_theta_cc_name(name):
            x[i] = min(float(ub[i]), float(x[i]) + shift)
            n_moved += 1
    opt.x = x
    print(f"DEBUG_INTERIOR_THETA: moved {n_moved} thetas_orientation_cc by +{shift}")
    return x


def preflight_is_clean(payload, hinge_keys=("Jbsd", "Jbca", "Jbcd")) -> bool:
    """True if selected hinge dJs, optional fun(x0), and stacked QR are finite/ok.

    ``hinge_keys`` should match the nonlinear constraints actually passed to
    ``minimize`` so an ablation that drops hinges is not aborted by them.
    """
    for key in hinge_keys:
        block = payload.get(key)
        if not block:
            continue
        if block.get("n_nan", 0) or block.get("n_inf", 0):
            return False
    fun_x0 = payload.get("fun_x0")
    if fun_x0 is not None:
        if fun_x0.get("n_nan", 0) or fun_x0.get("n_inf", 0):
            return False
        if fun_x0.get("J") in {"nan", "inf", "-inf"}:
            return False
    qr = payload.get("stacked_qr")
    if qr is not None and not qr.get("ok", False):
        return False
    return True


def build_preflight(
    *,
    Jbsd,
    Jbca,
    Jbcd,
    dofs,
    lb,
    ub,
    dof_names,
    linear_cons,
    include_jbcd=True,
    qr_hinges=None,
    fun_x0=None,
    extra=None,
) -> dict:
    payload = {
        "n_dofs": int(len(np.asarray(dofs))),
        "dof_names": list(dof_names),
        "bounds": bound_audit(dofs, lb, ub, dof_names),
        "sum_dphis": linear_sum_audit(linear_cons, dofs),
        "Jbsd": hinge_stats("Jbsd", Jbsd),
        "Jbca": hinge_stats("Jbca", Jbca),
        "Jbcd": hinge_stats("Jbcd", Jbcd),
        "Jbsd_shortest": _to_jsonable(float(Jbsd.shortest_distance())),
        "Jbca_smallest_angle_rad": _to_jsonable(float(Jbca.smallest_angle())),
        "Jbca_smallest_angle_deg": _to_jsonable(float(np.degrees(Jbca.smallest_angle()))),
        "Jbcd_shortest": _to_jsonable(float(Jbcd.shortest_distance())),
        "Jbcd_geometry": jbcd_geometry_stats(Jbcd),
    }
    if qr_hinges is None:
        qr_hinges = [Jbsd, Jbca]
        if include_jbcd:
            qr_hinges.append(Jbcd)
    payload["stacked_qr"] = stacked_constraint_qr(
        linear_cons,
        [np.asarray(obj.dJ(), dtype=float) for obj in qr_hinges],
    )
    payload["qr_hinge_names"] = [
        getattr(obj, "__class__", type(obj)).__name__ for obj in qr_hinges
    ]
    if fun_x0 is not None:
        payload["fun_x0"] = fun_x0
    if extra:
        payload.update(extra)
    return payload
