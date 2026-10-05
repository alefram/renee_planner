#!/usr/bin/env python3
"""Lexicographic HQP cascade solver.

A level minimizes ``||W (A v - b)||^2`` (least-squares) and/or enforces
``lower <= C v <= upper`` (inequality). Levels are solved in priority order;
once a level is solved, its achieved task value is frozen as an equality
constraint (within ``equality_tolerance``) for every subsequent level, and
its inequality rows remain in force for every subsequent level too. This is
the classic HQP technique (Kanoun et al.) applied at the velocity level.

``equality_tolerance`` must stay looser than OSQP's own ``eps_abs``/
``eps_rel`` (default 1e-5): a higher-priority level's returned solution is
only guaranteed to sit within its own bounds up to that solve tolerance, so
freezing it tighter than that can make the next level's QP infeasible
against the very bounds the frozen value was solved under -- this bites
hardest exactly when several inequalities are simultaneously active (e.g. a
saturated velocity limit), which is also when it matters least.

OSQP does the numerical work; ``hqp()`` itself is a pure function of arrays
and never touches ROS or ``RobotModel``, so it can be tested with synthetic
matrices.
"""

import contextlib
import os
import time
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import scipy.sparse as sparse


@contextlib.contextmanager
def _suppress_native_stdout():
    """Redirect OS-level fd 1 (stdout) to /dev/null for the wrapped block.

    OSQP's polish routine writes e.g. "Polishing not needed..." straight to
    the C stdout stream, bypassing the Python-level `verbose` setting, which
    floods the node's log at the control rate. Disabling
    polish instead of suppressing this would leave OSQP's returned solution
    up to ``eps_abs``/``eps_rel`` outside the active bounds -- enough for the
    next HQP level's frozen equality (see ``hqp()``) to become infeasible
    against the original inequality it was solved under.
    """
    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(devnull_fd)
        os.close(saved_fd)


@dataclass
class Level:
    name: str
    kind: Literal["equality", "least_squares"]
    A: np.ndarray
    b: np.ndarray
    weight: object = 1.0
    C: np.ndarray = None
    lower: np.ndarray = None
    upper: np.ndarray = None
    regularization: float = 1e-8


@dataclass
class LevelReport:
    name: str
    status: str
    residual_norm: float
    solution_norm: float
    active_inequalities: int
    solve_time_s: float


@dataclass
class Command:
    base_twist: np.ndarray
    arm_dq: np.ndarray
    arm_q_target: np.ndarray


def _weight_vector(weight, m: int) -> np.ndarray:
    if np.isscalar(weight):
        return np.full(m, float(weight))
    vector = np.asarray(weight, dtype=float)
    if vector.shape != (m,):
        raise ValueError(f"weight must be a scalar or shape ({m},), got {vector.shape}")
    return vector


def _validate(levels: Sequence[Level], nv: int) -> None:
    if not levels:
        raise ValueError("At least one level is required")
    for level in levels:
        if level.A is None or level.b is None:
            raise ValueError(f"Level '{level.name}': A and b are required (use empty arrays)")
        if level.A.ndim != 2 or level.A.shape[1] != nv:
            raise ValueError(f"Level '{level.name}': A must have shape (*, {nv})")
        if level.b.shape != (level.A.shape[0],):
            raise ValueError(f"Level '{level.name}': b must have shape ({level.A.shape[0]},)")
        if not np.all(np.isfinite(level.A)) or not np.all(np.isfinite(level.b)):
            raise ValueError(f"Level '{level.name}': A/b must be finite")
        if level.A.shape[0]:
            weight = _weight_vector(level.weight, level.A.shape[0])
            if np.any(weight <= 0):
                raise ValueError(f"Level '{level.name}': weight must be strictly positive")
        if level.regularization < 0:
            raise ValueError(f"Level '{level.name}': regularization must be non-negative")
        has_c = level.C is not None
        has_bounds = level.lower is not None or level.upper is not None
        if has_c != has_bounds:
            raise ValueError(f"Level '{level.name}': C requires both lower and upper bounds")
        if has_c:
            if level.C.ndim != 2 or level.C.shape[1] != nv:
                raise ValueError(f"Level '{level.name}': C must have shape (*, {nv})")
            m = level.C.shape[0]
            if level.lower.shape != (m,) or level.upper.shape != (m,):
                raise ValueError(f"Level '{level.name}': lower/upper must have shape ({m},)")
            if np.any(level.lower > level.upper):
                raise ValueError(f"Level '{level.name}': lower bound exceeds upper bound")


def _solve_qp(P: np.ndarray, q: np.ndarray, C: np.ndarray,
              lower: np.ndarray, upper: np.ndarray, *,
              warm_start: np.ndarray = None, osqp_settings: dict = None):
    """Wrapper around OSQP: minimize 0.5 x'Px + q'x s.t. lower <= Cx <= upper."""
    import osqp

    settings = {
        "verbose": False,
        # Polishing projects the solution back onto the active set, which
        # matters here: the next HQP level freezes this solve's achieved
        # value against the *same* bounds this level was solved under (see
        # hqp()), so an unpolished, eps_abs/eps_rel-inexact solution can
        # freeze a value that is technically outside its own bounds and make
        # the next level's QP infeasible. Its stdout chatter is suppressed
        # below instead of turning it off.
        "polish": True,
        "eps_abs": 1e-5,
        "eps_rel": 1e-5,
        "max_iter": 4000,
    }
    if osqp_settings:
        settings.update(osqp_settings)

    solver = osqp.OSQP()
    solver.setup(
        P=sparse.csc_matrix(np.asarray(P, dtype=float)),
        q=np.asarray(q, dtype=float),
        A=sparse.csc_matrix(np.asarray(C, dtype=float)),
        l=np.asarray(lower, dtype=float),
        u=np.asarray(upper, dtype=float),
        **settings,
    )
    if warm_start is not None:
        solver.warm_start(x=np.asarray(warm_start, dtype=float))
    with _suppress_native_stdout():
        result = solver.solve(raise_error=False)
    return result.x, result.info.status, result.info.obj_val


def hqp(levels: Sequence[Level], nv: int, *,
        equality_tolerance: float = 1e-4,
        osqp_settings: dict = None):
    """Solve the lexicographic QP cascade. Pure function: no ROS, no robot."""
    _validate(levels, nv)

    x = np.zeros(nv)
    accumulated_C = np.zeros((0, nv))
    accumulated_lower = np.zeros(0)
    accumulated_upper = np.zeros(0)
    reports = []

    for level in levels:
        start = time.perf_counter()
        m = level.A.shape[0]
        if m:
            weight = _weight_vector(level.weight, m)
            weighted_A = level.A * weight[:, None]
            P = level.A.T @ weighted_A + level.regularization * np.eye(nv)
            q_vec = -(weighted_A.T @ level.b)
        else:
            P = level.regularization * np.eye(nv)
            q_vec = np.zeros(nv)

        C_own = level.C if level.C is not None else np.zeros((0, nv))
        lower_own = level.lower if level.lower is not None else np.zeros(0)
        upper_own = level.upper if level.upper is not None else np.zeros(0)
        C_total = np.vstack([accumulated_C, C_own])
        lower_total = np.concatenate([accumulated_lower, lower_own])
        upper_total = np.concatenate([accumulated_upper, upper_own])

        x_sol, status, _ = _solve_qp(P, q_vec, C_total, lower_total, upper_total,
                                      warm_start=x, osqp_settings=osqp_settings)
        solve_time = time.perf_counter() - start

        if x_sol is None or status not in ("solved", "solved inaccurate"):
            reports.append(LevelReport(level.name, status or "error",
                                        float("nan"), float("nan"), 0, solve_time))
            break

        x = x_sol
        residual_norm = float(np.linalg.norm(level.A @ x - level.b)) if m else 0.0
        if C_total.shape[0]:
            values = C_total @ x
            active = int(np.sum((values - lower_total < equality_tolerance) |
                                 (upper_total - values < equality_tolerance)))
        else:
            active = 0
        reports.append(LevelReport(level.name, "solved", residual_norm,
                                    float(np.linalg.norm(x)), active, solve_time))

        accumulated_C, accumulated_lower, accumulated_upper = C_total, lower_total, upper_total
        if m:
            achieved = level.A @ x
            accumulated_C = np.vstack([accumulated_C, level.A])
            accumulated_lower = np.concatenate([accumulated_lower, achieved - equality_tolerance])
            accumulated_upper = np.concatenate([accumulated_upper, achieved + equality_tolerance])

    return x, reports


class HQPController:
    """Orchestrates RobotModel + tasks + the hqp() cascade into a Command."""

    def __init__(self, robot, level_tasks: dict, *,
                 equality_tolerance: float = 1e-4, osqp_settings: dict = None) -> None:
        """Store the robot, task hierarchy and solver settings used by step().

        Input:
            robot: RobotModel providing whole-body state and kinematics.
            level_tasks: ordered {level_name: [Task, ...]} dict, highest
                priority first.
            equality_tolerance: tolerance forwarded to hqp() on every step().
            osqp_settings: OSQP settings forwarded to hqp() on every step().

        Output:
            None.
        """
        self.robot = robot
        self.level_tasks = level_tasks
        self.equality_tolerance = equality_tolerance
        self.osqp_settings = osqp_settings

    def _assemble_level(self, name: str, tasks, dt: float) -> Level:
        """Build one solver.Level by stacking every task's contribution.

        Input:
            name: label for the resulting level.
            tasks: list of Task instances belonging to this level.
            dt: control period, forwarded to each task's build(robot, dt).

        Output:
            Level: A/b/weight rows from least-squares tasks, C/lower/upper
                rows from inequality tasks, and the largest regularization
                any task in this level requested.
        """
        nv = self.robot.nv
        a_rows, b_rows, w_rows = [], [], []
        c_rows, lower_rows, upper_rows = [], [], []
        regularization = None
        for task in tasks:
            if not task.active:
                continue
            block = task.build(self.robot, dt)
            if block.A is not None:
                a = np.atleast_2d(block.A)
                a_rows.append(a)
                b_rows.append(np.atleast_1d(block.b))
                w_rows.append(_weight_vector(block.weight if block.weight is not None else 1.0,
                                              a.shape[0]))
            if block.C is not None:
                c_rows.append(np.atleast_2d(block.C))
                lower_rows.append(np.atleast_1d(block.lower))
                upper_rows.append(np.atleast_1d(block.upper))
            if block.regularization is not None:
                regularization = (block.regularization if regularization is None
                                   else max(regularization, block.regularization))

        A = np.vstack(a_rows) if a_rows else np.zeros((0, nv))
        b = np.concatenate(b_rows) if b_rows else np.zeros(0)
        weight = np.concatenate(w_rows) if w_rows else 1.0
        C = np.vstack(c_rows) if c_rows else None
        lower = np.concatenate(lower_rows) if lower_rows else None
        upper = np.concatenate(upper_rows) if upper_rows else None
        level_kwargs = {} if regularization is None else {"regularization": regularization}
        return Level(name=name, kind="least_squares" if a_rows else "equality",
                     A=A, b=b, weight=weight, C=C, lower=lower, upper=upper, **level_kwargs)

    def step(self, dt: float):
        """Solve the cascade once and advance the robot's stored state.

        Input:
            dt: control period to solve and integrate over.

        Output:
            tuple[Command, list[LevelReport]]: Command with base twist, arm
                joint velocities and integrated arm joint position target;
                LevelReports from hqp(), one per level actually solved (may
                stop short of every level if one fails).

        Side effect:
            Advances self.robot's stored state by integrating the solved
            velocity over dt.
        """
        levels = [self._assemble_level(name, tasks, dt)
                  for name, tasks in self.level_tasks.items()]
        v, reports = hqp(levels, self.robot.nv,
                          equality_tolerance=self.equality_tolerance,
                          osqp_settings=self.osqp_settings)

        q_next = self.robot.integrate(v, dt)
        self.robot.update(q=q_next, dq=v)

        command = Command(
            base_twist=v[self.robot.base_v_slice].copy(),
            arm_dq=v[self.robot.arm_v_slice].copy(),
            arm_q_target=self.robot.q[self.robot.arm_q_slice].copy(),
        )
        return command, reports
