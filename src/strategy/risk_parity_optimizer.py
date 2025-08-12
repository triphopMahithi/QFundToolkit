import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.optimize import minimize


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
    print("Objective value at solution:", risk_parity_objective(result.x, cov))

    return result.x

# === ส่วนโหลดข้อมูลหุ้นจริง ===

def get_stock_returns(tickers, period_days=365):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_days)

    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna()
    return returns

# === MAIN ===

tickers = ["MSTR","DDOG", "JEPQ", "SGOV", "NVDA", "QBTS","BTC-USD"]  
returns = get_stock_returns(tickers, period_days=365)
cov = returns.cov().values  

weights = solve_risk_parity(cov)
rc = risk_contributions(weights, cov)
rc_pct = rc / np.sum(rc)

for i, (ticker, w, r) in enumerate(zip(tickers, weights, rc_pct)):
    print(f"{ticker:>6}: Weight = {w:.4f}, RC % = {r:.4f}")
