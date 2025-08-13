# -*- coding: utf-8 -*-
"""
Modern Portfolio Theory (MPT) – Formal, Adaptive Long-only, and CML
Demo universe: NVDA, DDOG, MSTR (monthly)
Requirements: numpy, pandas, matplotlib, yfinance
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.colors as mcolors
from adaptive_pgd import *
# ============================== Core Estimation ==============================

def calculate_mean_covariance(returns: np.ndarray, ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Calculate mean return vector and covariance matrix.

    Args:
        returns (np.ndarray): Asset returns, shape (T, N). Use same period for all series.
        ddof (int): Delta degrees of freedom for covariance. Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray]: (mean_returns, covariance_matrix)

    Example:
        >>> import numpy as np
        >>> rets = np.array([[0.01, 0.02], [0.03, 0.01], [0.02, 0.04]])
        >>> mu, Sigma = calculate_mean_covariance(rets)
        >>> mu.shape, Sigma.shape
        ((2,), (2, 2))
    """
    mean_returns = np.nanmean(returns, axis=0)
    covariance_matrix = np.cov(returns.T, ddof=ddof)
    return mean_returns, covariance_matrix


def invert_covariance_matrix(cov_matrix: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Invert covariance matrix with ridge stabilization.

    Args:
        cov_matrix (np.ndarray): Covariance matrix Σ, shape (N, N).
        ridge (float): Small ridge added to diagonal. Defaults to 1e-8.

    Returns:
        np.ndarray: Inverted covariance matrix Σ⁻¹.

    Example:
        >>> import numpy as np
        >>> Sigma = np.array([[0.04, 0.01],[0.01, 0.03]])
        >>> inv = invert_covariance_matrix(Sigma)
        >>> inv.shape
        (2, 2)
    """
    adjusted = cov_matrix + np.eye(cov_matrix.shape[0]) * ridge
    try:
        return np.linalg.inv(adjusted)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(adjusted)


def compute_abc_constants(mean_returns: np.ndarray, cov_matrix: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    """Compute Markowitz constants A, B, C and Delta.

    A = 1' Σ⁻¹ 1, B = 1' Σ⁻¹ μ, C = μ' Σ⁻¹ μ, Delta = AC - B²

    Args:
        mean_returns (np.ndarray): Mean return vector μ, shape (N,).
        cov_matrix (np.ndarray): Covariance matrix Σ, shape (N, N).

    Returns:
        tuple[float, float, float, float, np.ndarray]: (A, B, C, Delta, inv_cov)

    Example:
        >>> import numpy as np
        >>> mu = np.array([0.01, 0.02])
        >>> Sigma = np.array([[0.04, 0.01],[0.01, 0.03]])
        >>> A, B, C, Delta, inv = compute_abc_constants(mu, Sigma)
        >>> Delta > 0
        True
    """
    n = mean_returns.shape[0]
    ones = np.ones(n)
    inv_cov = invert_covariance_matrix(cov_matrix)
    A = ones @ inv_cov @ ones
    B = ones @ inv_cov @ mean_returns
    C = mean_returns @ inv_cov @ mean_returns
    Delta = A * C - B**2
    return A, B, C, Delta, inv_cov


# =================== Unrestricted (Long–Short Allowed) ======================

def compute_weights_unrestricted_target_return(mean_returns: np.ndarray,
                                               cov_matrix: np.ndarray,
                                               target_return: float) -> np.ndarray:
    """Closed-form weights for a given target return R (unrestricted, long–short).

    w(R) = Σ⁻¹ [ ((C - B R)/Δ)·1 + ((A R - B)/Δ)·μ ]

    Args:
        mean_returns (np.ndarray): Mean return vector μ, shape (N,).
        cov_matrix (np.ndarray): Covariance matrix Σ, shape (N, N).
        target_return (float): Target expected return R (per period).

    Returns:
        np.ndarray: Portfolio weights summing to 1 (can include negatives).

    Raises:
        ValueError: If Δ <= 0, indicating degenerate frontier.

    Example:
        >>> mu = np.array([0.01, 0.015, 0.012])
        >>> Sigma = 0.02*np.eye(3)
        >>> w = compute_weights_unrestricted_target_return(mu, Sigma, target_return=0.013)
        >>> abs(w.sum() - 1) < 1e-8
        True
    """
    A, B, C, Delta, inv_cov = compute_abc_constants(mean_returns, cov_matrix)
    if Delta <= 0:
        raise ValueError("Delta <= 0; check mean_returns/cov_matrix.")
    ones = np.ones_like(mean_returns)
    w = inv_cov @ ( ((C - B*target_return)/Delta)*ones + ((A*target_return - B)/Delta)*mean_returns )
    return w


def compute_weights_unrestricted_gmv(mean_returns: np.ndarray,
                                     cov_matrix: np.ndarray) -> np.ndarray:
    """Global Minimum Variance weights (unrestricted).

    w_GMV = Σ⁻¹ 1 / (1' Σ⁻¹ 1)

    Args:
        mean_returns (np.ndarray): Mean return vector μ (unused, kept for API symmetry).
        cov_matrix (np.ndarray): Covariance matrix Σ.

    Returns:
        np.ndarray: GMV weights (sum to 1, can include negatives).

    Example:
        >>> mu = np.array([0.01, 0.02, 0.015])
        >>> Sigma = 0.03*np.eye(3)
        >>> w = compute_weights_unrestricted_gmv(mu, Sigma)
        >>> abs(w.sum() - 1) < 1e-8
        True
    """
    A, _, _, _, inv_cov = compute_abc_constants(mean_returns, cov_matrix)
    ones = np.ones_like(mean_returns)
    return (inv_cov @ ones) / A


def compute_weights_unrestricted_tangency(mean_returns: np.ndarray,
                                          cov_matrix: np.ndarray,
                                          risk_free: float = 0.0) -> np.ndarray:
    """Tangency (max Sharpe) portfolio weights (unrestricted).

    w ∝ Σ⁻¹ (μ - r_f 1), then normalized to sum to 1.

    Args:
        mean_returns (np.ndarray): Mean return vector μ.
        cov_matrix (np.ndarray): Covariance matrix Σ.
        risk_free (float): Risk-free rate per period r_f. Defaults to 0.0.

    Returns:
        np.ndarray: Tangency weights (can include negatives).

    Raises:
        ValueError: If normalization denominator ~ 0 (degenerate case).
    """
    n = mean_returns.shape[0]
    ones = np.ones(n)
    inv_cov = invert_covariance_matrix(cov_matrix)
    excess = mean_returns - risk_free * ones
    raw = inv_cov @ excess
    denom = ones @ raw
    if abs(denom) < 1e-12:
        raise ValueError("Degenerate tangency: check μ, Σ, r_f.")
    return raw / denom

# ============================= Public Interface =============================

def optimize_mpt(mean_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 objective: str = "tangency",
                 mode: str = "unrestricted",
                 risk_free: float = 0.0,
                 target_return: float | None = None,
                 steps: int = 12000,
                 lr: float = 0.01,
                 w0: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Unified MPT optimizer (unrestricted vs long-only; tangency/GMV/target-return).

    Args:
        mean_returns (np.ndarray): Mean return vector μ, shape (N,).
        cov_matrix (np.ndarray): Covariance matrix Σ, shape (N, N).
        objective (str): {"tangency", "min_variance", "target_return"}.
        mode (str): {"unrestricted", "long_only"}.
        risk_free (float): Risk-free r_f per period.
        target_return (float | None): Required if objective == "target_return".
        steps (int): Iterations for long-only solvers.
        lr (float): Learning rate for long-only solvers.
        w0 (np.ndarray | None): Warm start for long-only solvers.

    Returns:
        tuple[np.ndarray, dict]: (weights, stats) where stats has keys:
            - expected_return
            - volatility
            - sharpe
    """
    mu = np.asarray(mean_returns).reshape(-1)
    Sigma = np.asarray(cov_matrix)

    if mode == "unrestricted":
        if objective == "tangency":
            w = compute_weights_unrestricted_tangency(mu, Sigma, risk_free)
        elif objective == "min_variance":
            w = compute_weights_unrestricted_gmv(mu, Sigma)
        elif objective == "target_return":
            if target_return is None:
                raise ValueError("target_return must be provided for objective='target_return'.")
            w = compute_weights_unrestricted_target_return(mu, Sigma, target_return)
        else:
            raise ValueError("Unknown objective.")
    elif mode == "long_only":
        if objective == "tangency":
            w = solve_max_sharpe_long_only(mu, Sigma, risk_free, w0=w0, steps=steps, lr=lr)
        elif objective == "min_variance":
            w = solve_min_variance_long_only_adaptive(Sigma, w0=w0, max_steps=steps)
        elif objective == "target_return":
            if target_return is None:
                raise ValueError("target_return must be provided for objective='target_return'.")
            w = solve_target_return_long_only(mu, Sigma, target_return, w0=w0, steps=steps, lr=lr)
        else:
            raise ValueError("Unknown objective.")
    else:
        raise ValueError("mode must be 'unrestricted' or 'long_only'.")

    ret = float(w @ mu)
    vol = float(np.sqrt(w @ Sigma @ w))
    sharpe = (ret - risk_free) / vol if vol > 0 else np.nan
    stats = dict(expected_return=ret, volatility=vol, sharpe=sharpe)
    return w, stats

# =================== Capital Market Line (CML) Utilities ====================

def compute_cml_points(mean_returns: np.ndarray,
                       cov_matrix: np.ndarray,
                       risk_free: float = 0.0,
                       n_points: int = 100,
                       sigma_max_factor: float = 1.3) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (sigmas, mus) on the Capital Market Line (CML).

    CML: E[R_p] = R_f + Sharpe_T * sigma_p, where Sharpe_T = (mu_T - R_f)/sigma_T

    Args:
        mean_returns (np.ndarray): Mean return vector μ.
        cov_matrix (np.ndarray): Covariance matrix Σ.
        risk_free (float): Risk-free rate r_f per period.
        n_points (int): Number of points on the CML.
        sigma_max_factor (float): Extend beyond σ_T to show leverage.

    Returns:
        tuple[np.ndarray, np.ndarray, dict]: (sigmas, mus, stats_dict)
            stats_dict includes {"mu_t", "sigma_t", "sharpe_t"}.
    """
    w_tan = compute_weights_unrestricted_tangency(mean_returns, cov_matrix, risk_free)
    mu_t = float(w_tan @ mean_returns)
    sigma_t = float(np.sqrt(w_tan @ cov_matrix @ w_tan))
    sharpe_t = (mu_t - risk_free) / sigma_t
    sigmas = np.linspace(0.0, sigma_t * sigma_max_factor, n_points)
    mus = risk_free + sharpe_t * sigmas
    return sigmas, mus, dict(mu_t=mu_t, sigma_t=sigma_t, sharpe_t=sharpe_t)



# ======================= Academic Plotting Utilities ========================

def build_dense_frontier(mean_returns: np.ndarray,
                         cov_matrix: np.ndarray,
                         mode: str = "unrestricted",
                         n_points_dense: int = 400,
                         steps: int = 12000,
                         lr: float = 0.01) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Build a dense efficient frontier for smoother academic plots (with warm-start for long-only)."""
    mu = np.asarray(mean_returns).reshape(-1)
    r_min, r_max = mu.min(), mu.max()
    pad = 0.03 * (r_max - r_min if r_max > r_min else max(abs(r_min), 1e-6))
    grid = np.linspace(r_min - pad, r_max + pad, n_points_dense)

    vols, rets, ws = [], [], []
    w_prev = None
    for R in grid:
        try:
            if mode == "unrestricted":
                w, _ = optimize_mpt(mu, cov_matrix, objective="target_return",
                                    mode="unrestricted", target_return=R)
            else:
                w, _ = optimize_mpt(mu, cov_matrix, objective="target_return",
                                    mode="long_only", target_return=R, steps=steps, lr=lr, w0=w_prev)
            vols.append(float(np.sqrt(w @ cov_matrix @ w)))
            rets.append(float(w @ mu))
            ws.append(w)
            w_prev = w
        except Exception:
            continue

    vols = np.array(vols); rets = np.array(rets)
    order = np.argsort(vols)
    return vols[order], rets[order], [ws[i] for i in order]

# =============================== Demo Runner ================================

if __name__ == "__main__":
    # --- Data: NVDA, DDOG, MSTR (monthly) ---
    tickers = ["NVDA", "DDOG", "MSTR"]
    start_date = "2018-01-01"

    prices = yf.download(tickers, start=start_date, auto_adjust=True)["Close"].dropna()
    # Monthly log returns
    daily_log = np.log(prices / prices.shift(1)).dropna()
    monthly_log = daily_log.resample("M").sum().dropna()

    mu, Sigma = calculate_mean_covariance(monthly_log.values)
    print("Tickers:", tickers)
    print("Mean (monthly):", np.round(mu, 6))
    print("Cov (monthly) shape:", Sigma.shape)

    rf = 0.0  # set monthly risk-free if needed

    # Tangency (Unrestricted vs Long-only)
    w_u, s_u = optimize_mpt(mu, Sigma, objective="tangency", mode="unrestricted", risk_free=rf)
    # warm-start long-only tangency using previous solution (optional; here w0=None)
    w_l, s_l = optimize_mpt(mu, Sigma, objective="tangency", mode="long_only", risk_free=rf, steps=15000, lr=0.01)

    print("\n=== Tangency (Unrestricted) ===")
    print("weights:", dict(zip(tickers, np.round(w_u, 4))))
    print("stats:", {k: round(v, 4) for k, v in s_u.items()})

    print("\n=== Tangency (Long-only) ===")
    print("weights:", dict(zip(tickers, np.round(w_l, 4))))
    print("stats:", {k: round(v, 4) for k, v in s_l.items()})

    # GMV (Unrestricted vs Long-only)
    w_gmv_u, s_gmv_u = optimize_mpt(mu, Sigma, objective="min_variance", mode="unrestricted", risk_free=rf)
    w_gmv_l = solve_min_variance_long_only_adaptive(Sigma, w0=None, max_steps=20000)
    s_gmv_l = {
        "expected_return": float(w_gmv_l @ mu),
        "volatility": float(np.sqrt(w_gmv_l @ Sigma @ w_gmv_l)),
        "sharpe": (float(w_gmv_l @ mu) - rf) / (float(np.sqrt(w_gmv_l @ Sigma @ w_gmv_l)) + 1e-18),
    }

    print("\n=== GMV (Unrestricted) ===")
    print("weights:", dict(zip(tickers, np.round(w_gmv_u, 4))))
    print("stats:", {k: round(v, 6) for k, v in s_gmv_u.items()})

    print("\n=== GMV (Long-only, Adaptive) ===")
    print("weights:", dict(zip(tickers, np.round(w_gmv_l, 4))))
    print("stats:", {k: round(v, 6) for k, v in s_gmv_l.items()})