from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from typing import Sequence, Dict, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
def portfolio_vol(w, cov):
    return np.sqrt(w @ cov @ w)

def risk_contributions(w, cov):
    total_vol = portfolio_vol(w, cov)
    mrc = cov @ w
    rc = w * mrc / total_vol
    return rc

def risk_parity_objective(w, cov):
    rc = risk_contributions(w, cov)
    return np.sum((rc - np.mean(rc))**2)

def solve_risk_parity(cov):
    n = cov.shape[0]
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    result = minimize(
        risk_parity_objective,
        x0,
        args=(cov,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        print("Optimization failed:", result.message)
        return x0
    logging.info("Objective value at solution: {}".format(round(risk_parity_objective(result.x, cov), 6)))
    return result.x


def get_stock_returns(tickers, period_days=365):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_days)

    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    returns = data.pct_change().dropna()
    return returns

def risk_parity_weights(
    tickers: Sequence[str],
    returns: pd.DataFrame,
    *,
    include_rc: bool = False,
    normalize: bool = True,
    logger: Optional[logging.Logger] = None,
    verbose: bool = False,
) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute Risk Parity portfolio weights and return them as a dict.

    Parameters
    ----------
    tickers : Sequence[str]
        Asset symbols. If returns has matching column names, order will follow `tickers`.
    returns : pd.DataFrame
        Asset return series; columns should correspond to `tickers`.
    include_rc : bool, default False
        If True, include each asset's risk-contribution share (0–1) in the output.
    normalize : bool, default True
        If True, rescale weights to sum to 1 (guarding against numeric drift).
    logger : logging.Logger, optional
        If provided, logs a compact table instead of printing.

    Returns
    -------
    dict
        - If include_rc=False: {ticker: weight}
        - If include_rc=True:  {ticker: {"weight": w, "rc_share": s}}

    Raises
    ------
    ValueError
        If dimensions mismatch, NaNs in covariance, or invalid solver output.
    """
    if len(tickers) == 0:
        raise ValueError("tickers must be non-empty.")
    if returns.shape[1] < len(tickers):
        raise ValueError("returns must have at least as many columns as tickers.")

    tickers = list(tickers)
    if set(tickers).issubset(set(returns.columns)):
        returns = returns[tickers]  # enforce order
    elif returns.shape[1] != len(tickers):
        raise ValueError("Length of tickers must match number of return columns.")

    if len(set(tickers)) != len(tickers):
        raise ValueError("Duplicate tickers detected.")

    cov = returns.cov(min_periods=1).to_numpy(dtype=float)
    if not np.isfinite(cov).all():
        raise ValueError("Covariance contains NaN/inf; clean or impute your data first.")

    w = np.asarray(solve_risk_parity(cov), dtype=float).reshape(-1)
    if w.shape[0] != len(tickers) or not np.isfinite(w).all():
        raise ValueError("Invalid weights returned by solver.")
    if normalize:
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            raise ValueError("Sum of weights is non-positive/invalid.")
        w = w / s

    rc = np.asarray(risk_contributions(w, cov), dtype=float).reshape(-1)
    if not np.isfinite(rc).all():
        raise ValueError("Risk contributions contain NaN/inf.")
    rc_total = rc.sum()
    if rc_total <= 0 or not np.isfinite(rc_total):
        raise ValueError("Total risk contribution is non-positive/invalid.")
    rc_share = rc / rc_total  # 0–1

    # --- Build result dict
    if include_rc:
        result: Dict[str, Dict[str, float]] = {
            t: {"weight": float(wi), "rc_share": float(si)}
            for t, wi, si in zip(tickers, w, rc_share)
        }
    else:
        result: Dict[str, float] = {t: float(wi) for t, wi in zip(tickers, w)}

    # --- Optional logging
    if logger:
        df_log = pd.DataFrame({
            "ticker": tickers,
            "weight": w,
            "rc_share": rc_share
        })
        logger.debug("Risk parity result:\n%s", df_log.round(6).to_string(index=False))

    # --- Optional verbose
    if verbose:
        for i, (ticker, weight, r) in enumerate(zip(tickers, w, rc_share)):
            print(f"{ticker:>6}: Weight = {weight:.4f}, RC % = {r:.4f}")
    return result


# === MAIN ===

tickers = ["MSTR","DDOG","NVDA", "JEPQ", "SGOV", "QBTS","QQQ"]  
returns = get_stock_returns(tickers, period_days=365)

risk_parity = risk_parity_weights(tickers, returns)
print(risk_parity)