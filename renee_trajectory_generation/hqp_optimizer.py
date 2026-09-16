"""Small lexicographic QP solver and sequential trajectory optimizer.

All vectors use SI units. A task is ``(A, b)`` and minimizes
``0.5 * ||A @ x - b||**2``. Constraints use ``lower <= C @ x <= upper``.
ROS and robot-specific calculations deliberately stay outside this module.
"""

import numpy as np
from scipy import sparse


def solve_hqp(tasks, constraints, options=None):
    """Solve ordered least-squares tasks without trading higher priorities.

    Parameters
    ----------
    tasks : sequence of (array_like, array_like)
        Ordered matrices A (m, n) and target vectors b (m,).
    constraints : tuple
        C (r, n), lower (r,), upper (r,); equal bounds express equalities.
    options : dict, optional
        ``constraint_tolerance`` and ``qp_max_iterations``.

    Returns
    -------
    dict
        Status, solution (or None), per-level residual norms and message.

    Raises
    ------
    ValueError
        Inconsistent dimensions, nonfinite task data or invalid bounds.
    """
    import osqp

    options = options or {}
    tolerance = float(options.get('constraint_tolerance', 1e-6))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('constraint_tolerance must be positive and finite')
    if not tasks:
        raise ValueError('At least one task is required')
    c, lower, upper = constraints
    c = sparse.csc_matrix(c, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (lower.shape != (c.shape[0],) or upper.shape != lower.shape or
            np.any(np.isnan(lower)) or np.any(np.isnan(upper)) or
            np.any(lower > upper) or not np.all(np.isfinite(c.data)) or
            np.any(np.isposinf(lower)) or np.any(np.isneginf(upper))):
        raise ValueError('Invalid constraint matrix or bounds')
    levels = []
    for matrix, target in tasks:
        matrix = sparse.csc_matrix(matrix, dtype=float)
        target = np.asarray(target, dtype=float)
        if (matrix.shape[1] != c.shape[1] or target.shape != (matrix.shape[0],) or
                not np.all(np.isfinite(matrix.data)) or
                not np.all(np.isfinite(target))):
            raise ValueError('Invalid task dimensions or values')
        levels.append((matrix, target))

    residuals, locked = [], []
    x = np.zeros(c.shape[1])
    for matrix, target in levels:
        solver = osqp.OSQP()
        # No epsilon*I: that would change the optimum of a higher priority.
        solver.setup(P=sparse.triu(matrix.T @ matrix, format='csc'),
                     q=np.asarray(-matrix.T @ target).ravel(), A=c,
                     l=lower, u=upper, verbose=False, polishing=False,
                     eps_abs=tolerance * 0.1, eps_rel=tolerance * 0.1,
                     max_iter=int(options.get('qp_max_iterations', 10000)))
        solver.warm_start(x=x)
        result = solver.solve(raise_error=False)
        if result.info.status != 'solved' or result.x is None:
            return {'status': 'qp_failed', 'x': None, 'residuals': residuals,
                    'message': result.info.status}
        x = result.x
        values = c @ x
        violation = max(np.max(lower - values, initial=0.0),
                        np.max(values - upper, initial=0.0))
        if not np.all(np.isfinite(x)) or violation > tolerance:
            return {'status': 'numerical_failure', 'x': None,
                    'residuals': residuals, 'message': 'Constraint tolerance exceeded'}
        if any(np.max(np.abs(a @ x - y), initial=0.0) > tolerance
               for a, y in locked):
            return {'status': 'numerical_failure', 'x': None,
                    'residuals': residuals, 'message': 'Higher priority changed'}
        achieved = np.asarray(matrix @ x).ravel()
        residuals.append(float(np.linalg.norm(achieved - target)))
        locked.append((matrix, achieved))
        # The fitted vector is unique even for a rank-deficient least-squares
        # problem. Locking it leaves precisely the freedom of lower levels.
        c = sparse.vstack((c, matrix), format='csc')
        lower = np.concatenate((lower, achieved))
        upper = np.concatenate((upper, achieved))
    return {'status': 'solved', 'x': x, 'residuals': residuals, 'message': ''}


def finite_jacobian(function, x, step=1e-5):
    """Return a central-difference Jacobian; x and function output are flat."""
    x = np.asarray(x, dtype=float)
    columns = []
    for index in range(x.size):
        offset = np.zeros_like(x)
        offset[index] = step
        columns.append((function(x + offset) - function(x - offset)) / (2 * step))
    return np.column_stack(columns)


def _violation(values, lower, upper):
    return float(max(np.max(lower - values, initial=0.0),
                     np.max(values - upper, initial=0.0)))


def _lexicographic_accept(before, after, tolerance):
    for old, new in zip(before, after):
        if new > old + tolerance:
            return False
        if new < old - tolerance:
            return True
    return True


def optimize_scan(initial, residual_functions, constraint_function, options=None,
                  valid_function=None):
    """Optimize a trajectory with sequential HQP and nonlinear verification.

    Parameters
    ----------
    initial : ndarray, shape (samples, joints)
        Feasible joint seed, in radians. No robot-specific state is assumed.
    residual_functions : sequence of callable
        Each accepts a flattened trajectory and returns a residual vector.
    constraint_function : callable
        Returns nonlinear values, lower bounds and upper bounds. Its row
        ordering must remain fixed during optimization.
    options : dict, optional
        Iteration, trust-region, finite-difference and tolerance settings.
    valid_function : callable, optional
        Additional exact checks such as visibility; receives flat joints.

    Returns
    -------
    dict
        Status, joint trajectory, iterations, residuals and maximum violation.
        Only ``converged`` is a successful nonlinear result.
    """
    options = options or {}
    initial = np.asarray(initial, dtype=float)
    if initial.ndim != 2 or not initial.size or not np.all(np.isfinite(initial)):
        raise ValueError('initial must be a nonempty finite 2D trajectory')
    if not residual_functions:
        raise ValueError('At least one residual function is required')
    shape, x = initial.shape, initial.ravel().copy()
    tol = options.get('constraint_tolerance', 1e-6)
    step_tol = options.get('step_tolerance_rad', 1e-5)
    trust = options.get('trust_region_rad', 0.15)
    difference = options.get('finite_difference_step', 1e-5)
    max_iterations = options.get('max_iterations', 40)
    status, message, iteration = 'max_iterations', '', 0
    values, lower, upper = constraint_function(x)
    if _violation(values, lower, upper) > tol or (
            valid_function is not None and not valid_function(x)):
        status, message = 'invalid_seed', 'A feasible visible seed is required'
    else:
        for iteration in range(1, max_iterations + 1):
            residuals = [np.asarray(f(x)) for f in residual_functions]
            tasks = [(finite_jacobian(f, x, difference), -r)
                     for f, r in zip(residual_functions, residuals)]
            values, lower, upper = constraint_function(x)
            jac = finite_jacobian(lambda y: constraint_function(y)[0], x, difference)
            bounds = (sparse.vstack((sparse.csc_matrix(jac), sparse.eye(x.size))),
                      np.r_[lower - values, np.full(x.size, -trust)],
                      np.r_[upper - values, np.full(x.size, trust)])
            qp = solve_hqp(tasks, bounds, options)
            if qp['status'] != 'solved':
                status, message = qp['status'], qp['message']
                break
            delta = qp['x']
            before = [float(r @ r) for r in residuals]
            accepted = False
            for reduction in range(options.get('line_search_steps', 12)):
                candidate = x + delta * (0.5 ** reduction)
                v, lo, hi = constraint_function(candidate)
                after = [float(np.linalg.norm(f(candidate)) ** 2)
                         for f in residual_functions]
                if (_violation(v, lo, hi) <= tol and
                        _lexicographic_accept(before, after, tol) and
                        (valid_function is None or valid_function(candidate))):
                    accepted = True
                    break
            if not accepted:
                trust *= 0.5
                if trust < step_tol:
                    status, message = 'stalled', 'No verified improving step'
                    break
                continue
            x = candidate
            # A tiny line-search fraction alone is not convergence: the full
            # local HQP step must also be small.
            if np.max(np.abs(delta), initial=0.0) < step_tol:
                status = 'converged'
                break
    values, lower, upper = constraint_function(x)
    return {'status': status, 'message': message, 'joints': x.reshape(shape),
            'iterations': iteration,
            'residuals': [float(np.linalg.norm(f(x))) for f in residual_functions],
            'max_violation': _violation(values, lower, upper)}
