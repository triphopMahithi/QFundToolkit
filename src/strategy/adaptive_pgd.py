import numpy as np

# ========================== Long-Only (Adaptive PGD) ========================

def _objective_variance(w: np.ndarray, cov_matrix: np.ndarray) -> float:
    return float(w @ cov_matrix @ w)

def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * (np.arange(1, n+1)) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0.0)

def _power_iteration_lmax(cov_matrix: np.ndarray, iters: int = 8) -> float:
    n = cov_matrix.shape[0]
    x = np.ones(n) / np.sqrt(n)
    for _ in range(iters):
        x = cov_matrix @ x
        x = x / (np.linalg.norm(x) + 1e-18)
    lam = float(x @ cov_matrix @ x)
    return lam

def solve_min_variance_long_only_adaptive(
    cov_matrix: np.ndarray,
    w0: np.ndarray | None = None,
    max_steps: int = 20000,
    lr0: float | None = None,
    tol: float = 1e-10,
    armijo: bool = True,
    bb_step: bool = True,
    patience: int = 3,
) -> np.ndarray:
    """Projected Gradient for long-only GMV with BB step + Armijo + safe lr.

    Args:
        cov_matrix (np.ndarray): Covariance matrix Σ.
        w0 (np.ndarray | None): Warm start (optional).
        max_steps (int): Maximum iterations.
        lr0 (float | None): Initial learning rate; if None, use 0.9/(2*λ_max).
        tol (float): L1 stopping tolerance per iteration.
        armijo (bool): Use Armijo backtracking.
        bb_step (bool): Use Barzilai–Borwein step size update.
        patience (int): Stop if ||Δw||₁ < tol holds for this many consecutive steps.

    Returns:
        np.ndarray: Long-only GMV weights (sum to 1).

    Example:
        >>> Sigma = 0.05*np.eye(3)
        >>> w = solve_min_variance_long_only_adaptive(Sigma)
        >>> (w >= 0).all() and abs(w.sum()-1) < 1e-8
        True
    """
    n = cov_matrix.shape[0]
    w = _project_to_simplex(np.ones(n)/n if w0 is None else w0)
    grad = 2.0 * (cov_matrix @ w)
    f_prev = _objective_variance(w, cov_matrix)

    if lr0 is None:
        lam_max = _power_iteration_lmax(cov_matrix, iters=8)
        L = 2.0 * lam_max
        lr = 0.9 / (L + 1e-12)
    else:
        lr = float(lr0)

    no_improve = 0
    for _ in range(1, max_steps + 1):
        w_old, grad_old, f_old = w, grad, f_prev

        # tentative step
        w_tent = _project_to_simplex(w - lr * grad)

        # Armijo backtracking (sufficient decrease)
        if armijo:
            c = 1e-4
            step = 0
            f_tent = _objective_variance(w_tent, cov_matrix)
            while f_tent > f_prev - c * lr * np.dot(grad, (w - w_tent)) and step < 10:
                lr *= 0.5
                w_tent = _project_to_simplex(w - lr * grad)
                f_tent = _objective_variance(w_tent, cov_matrix)
                step += 1
        else:
            f_tent = _objective_variance(w_tent, cov_matrix)

        # accept
        w = w_tent
        f_prev = f_tent
        grad = 2.0 * (cov_matrix @ w)

        # Barzilai–Borwein step
        if bb_step:
            s = w - w_old
            y = grad - grad_old
            denom = float(y @ y) + 1e-18
            lr = max(1e-6, min(10.0, float((s @ y) / denom)))

        # stopping
        if np.linalg.norm(w - w_old, 1) < tol:
            no_improve += 1
        else:
            no_improve = 0
        if no_improve >= patience:
            break

    return w


def solve_max_sharpe_long_only(mean_returns: np.ndarray,
                               cov_matrix: np.ndarray,
                               risk_free: float = 0.0,
                               w0: np.ndarray | None = None,
                               steps: int = 15000,
                               lr: float = 0.01,
                               tol: float = 1e-10) -> np.ndarray:
    """Solve max-Sharpe (long-only) via Projected Gradient on -Sharpe.

    Maximize (w' (μ - r_f)) / sqrt(w' Σ w)  s.t. w in simplex.

    Args:
        mean_returns (np.ndarray): Mean return vector μ.
        cov_matrix (np.ndarray): Covariance matrix Σ.
        risk_free (float): Risk-free rate r_f per period.
        w0 (np.ndarray | None): Warm start.
        steps (int): Maximum iterations.
        lr (float): Learning rate.
        tol (float): L1 stopping tolerance.

    Returns:
        np.ndarray: Long-only tangency weights.
    """
    n = mean_returns.shape[0]
    ones = np.ones(n)
    excess = mean_returns - risk_free * ones
    w = _project_to_simplex(np.ones(n)/n if w0 is None else w0)
    eps = 1e-12

    no_improve = 0
    for _ in range(steps):
        num = float(w @ excess)
        den2 = float(w @ cov_matrix @ w) + eps
        den = np.sqrt(den2)
        grad_num = excess
        grad_den = (cov_matrix @ w) / den
        grad = -(grad_num * den - num * grad_den) / (den**2 + eps)
        w_new = _project_to_simplex(w - lr * grad)

        if np.linalg.norm(w_new - w, 1) < tol:
            no_improve += 1
            if no_improve >= 3:
                w = w_new
                break
        else:
            no_improve = 0

        w = w_new
    return w


def solve_target_return_long_only(mean_returns: np.ndarray,
                                  cov_matrix: np.ndarray,
                                  target_return: float,
                                  w0: np.ndarray | None = None,
                                  steps: int = 12000,
                                  lr: float = 0.01,
                                  tol: float = 1e-10,
                                  penalty_grid: np.ndarray | None = None,
                                  armijo: bool = True) -> np.ndarray:
    """Approximate min-variance at target return (long-only) via penalty + Armijo.

    Minimize  w' Σ w + λ (w' μ - R)^2  s.t. w in simplex, sweep λ.

    Args:
        mean_returns (np.ndarray): Mean return vector μ.
        cov_matrix (np.ndarray): Covariance matrix Σ.
        target_return (float): Target expected return R.
        w0 (np.ndarray | None): Warm start.
        steps (int): Max iterations per λ.
        lr (float): Learning rate.
        tol (float): L1 stopping tolerance.
        penalty_grid (np.ndarray | None): λ values to sweep. Defaults to logspace 1e-3..1e2.
        armijo (bool): Use Armijo backtracking on the penalty objective.

    Returns:
        np.ndarray: Approximate long-only weights at target return.
    """
    n = mean_returns.shape[0]
    w_init = _project_to_simplex(np.ones(n)/n if w0 is None else w0)
    if penalty_grid is None:
        penalty_grid = np.geomspace(1e-3, 1e2, num=5)

    best_w, best_obj = None, np.inf
    for lam in penalty_grid:
        w = w_init.copy()
        f_prev = np.inf
        no_improve = 0
        for _ in range(steps):
            diff = float(w @ mean_returns - target_return)
            grad = 2.0 * (cov_matrix @ w) + 2.0 * lam * diff * mean_returns

            # tentative step
            w_new = _project_to_simplex(w - lr * grad)

            # Armijo backtracking on penalty objective
            if armijo:
                c = 1e-4
                step = 0
                f_new = float(w_new @ cov_matrix @ w_new) + lam * diff**2
                while f_new > f_prev - c * lr * np.dot(grad, (w - w_new)) and step < 10:
                    lr *= 0.5
                    w_new = _project_to_simplex(w - lr * grad)
                    f_new = float(w_new @ cov_matrix @ w_new) + lam * diff**2
                    step += 1
            else:
                f_new = float(w_new @ cov_matrix @ w_new) + lam * diff**2

            if np.linalg.norm(w_new - w, 1) < tol:
                no_improve += 1
                if no_improve >= 3:
                    w = w_new
                    break
            else:
                no_improve = 0

            w, f_prev = w_new, f_new

        obj = float(w @ cov_matrix @ w) + lam * (float(w @ mean_returns - target_return))**2
        if obj < best_obj:
            best_obj, best_w = obj, w

    return best_w
