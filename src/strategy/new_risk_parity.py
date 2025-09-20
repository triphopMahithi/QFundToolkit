from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize
from typing import Sequence, Dict, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)

# ===================== Core math =====================

def portfolio_var(w: np.ndarray, cov: np.ndarray) -> float:
    return float(w @ cov @ w)

def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    v = portfolio_var(w, cov)
    return np.sqrt(v) if v > 0 else 0.0

def risk_contributions_share(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    RC (absolute) = w_i * (Σ w)_i
    RC (share)    = RC_abs / (w' Σ w)
    RC_share รวมกัน = 1
    """
    mrc = cov @ w                     # marginal risk to variance
    rc_abs = w * mrc                  # absolute contribution to variance
    denom = portfolio_var(w, cov)     # w' Σ w
    if denom <= 0 or not np.isfinite(denom):
        # ป้องกันกรณีพอร์ตแปลก ๆ
        return np.full_like(w, 1.0 / len(w))
    rc_share = rc_abs / denom
    # cleanup เชิงตัวเลขเล็กน้อย
    rc_share = np.clip(rc_share, 0.0, 1.0)
    s = rc_share.sum()
    return rc_share / s if s > 0 else np.full_like(w, 1.0 / len(w))

def risk_parity_objective(w: np.ndarray, cov: np.ndarray, target: Optional[np.ndarray] = None, l2: float = 0.0) -> float:
    """
    ทำให้ RC_share เข้าใกล้ target (เท่ากัน = 1/n)
    + L2 regularization เล็กน้อยเพื่อความนิ่ง
    """
    n = len(w)
    rc_share = risk_contributions_share(w, cov)
    if target is None:
        target = np.full(n, 1.0 / n)
    diff = rc_share - target
    obj = float(diff @ diff)
    if l2 > 0:
        obj += float(l2 * (w @ w))
    return obj

def solve_risk_parity(cov: np.ndarray, bounds=(0.0, 1.0), target: Optional[np.ndarray] = None, l2: float = 0.0):
    n = cov.shape[0]
    x0 = np.full(n, 1.0 / n)
    bnds = [bounds] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    res = minimize(
        risk_parity_objective,
        x0,
        args=(cov, target, l2),
        method='SLSQP',
        bounds=bnds,
        constraints=cons,
        options={'ftol': 1e-12, 'maxiter': 200}
    )
    if not res.success or not np.isfinite(res.x).all():
        logging.warning(f"Optimization failed: {res.message}. Using equal weight.")
        return x0
    logging.info("Objective value at solution: %.6g", risk_parity_objective(res.x, cov, target, l2))
    return res.x

# ===================== Data helper =====================

def get_stock_returns(tickers, period_days=365):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_days)
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    # ให้แน่ใจว่าเป็น DataFrame เสมอ
    if isinstance(data, pd.Series):
        data = data.to_frame()
    returns = data.pct_change().dropna(how='any')  # ตัดวันที่มี NaN ออกทั้งแถว
    return returns

# ===================== Public API =====================

def risk_parity_weights(
    tickers: Sequence[str],
    returns: pd.DataFrame,
    *,
    include_rc: bool = False,
    normalize: bool = True,
    logger: Optional[logging.Logger] = None,
    verbose: bool = False,
    weight_bounds=(0.0, 0.35),      # << ป้องกันการกองตัวเดียวเกินไป
    l2: float = 0.0,                # << ตั้ง 1e-4 ถ้าอยากให้กระจายนิ่งขึ้น
    target_share: Optional[np.ndarray] = None  # << กำหนด budget ไม่เท่ากันได้ เช่น GLD สูงขึ้น ฯลฯ
) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:

    if len(tickers) == 0:
        raise ValueError("tickers must be non-empty.")
    tickers = list(tickers)

    # บังคับคอลัมน์เรียงตาม tickers
    if not set(tickers).issubset(set(returns.columns)):
        raise ValueError("returns must contain all tickers as columns.")
    rets = returns[tickers].copy()

    # คอเวเรียนซ์ (ใช้ sample cov ตรง ๆ; ถ้าตลาดต่างเขตเวลา การ dropna ข้างบนช่วยให้ alignment ดี)
    cov = rets.cov(min_periods=1).to_numpy(dtype=float)
    if not np.isfinite(cov).all():
        raise ValueError("Covariance contains NaN/inf; clean data first.")

    w = np.asarray(solve_risk_parity(cov, bounds=weight_bounds, target=target_share, l2=l2), dtype=float).reshape(-1)
    if normalize:
        s = w.sum()
        w = w / s if s > 0 else np.full_like(w, 1.0 / len(w))

    rc_share = risk_contributions_share(w, cov)

    if include_rc:
        result: Dict[str, Dict[str, float]] = {
            t: {"weight": float(wi), "rc_share": float(si)}
            for t, wi, si in zip(tickers, w, rc_share)
        }
    else:
        result: Dict[str, float] = {t: float(wi) for t, wi in zip(tickers, w)}

    # logging / verbose
    if logger:
        df_log = pd.DataFrame({"ticker": tickers, "weight": w, "rc_share": rc_share})
        logger.debug("Risk parity result:\n%s", df_log.round(6).to_string(index=False))
    if verbose:
        for t, wi, si in zip(tickers, w, rc_share):
            print(f"{t:>6}: Weight = {wi:.4f}, RC % = {si:.4f}")
    return result

# ===================== MAIN =====================

tickers = ["NVDA","DDOG","JEPQ","UNH","RBLX","ABBV","INTC","META","GLD","SMR","UBER"]
returns = get_stock_returns(tickers, period_days=365*5)

# ตั้ง budget เท่ากัน (ถ้าต้องการให้ GLD รับเสี่ยงมากขึ้น สามารถตั้ง target_share เป็นเวกเตอร์เองได้)
risk_parity = risk_parity_weights(
    tickers,
    returns,
    include_rc=True,
    verbose=True,
    weight_bounds=(0.0, 0.35),  # ป้องกันน้ำหนักล้น
    l2=1e-6                     # เล็กน้อยเพื่อความนิ่ง (ตั้ง 0 ได้)
)
print(risk_parity)
