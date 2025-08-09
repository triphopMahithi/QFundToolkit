import numpy as np
import yfinance as yf
from typing import List, Dict

def get_log_returns(tickers: List[str], start: str, end: str):
    data = yf.download(tickers, start=start, end=end)['Close']
    data = data.dropna()
    log_ret = np.log(data / data.shift(1)).dropna()
    return log_ret.values, data.columns.tolist()

def kelly_objective(f: np.ndarray, R: np.ndarray):
    port_ret = R @ f
    if np.any(1 + port_ret <= 0):
        return np.inf
    return -np.mean(np.log(1 + port_ret))

def kelly_gradient(f: np.ndarray, R: np.ndarray):
    port_ret = R @ f
    if np.any(1 + port_ret <= 0):
        return np.full_like(f, np.nan)
    grad = -np.mean((R.T / (1 + port_ret)), axis=1)
    return grad

def project_to_constraints(f: np.ndarray):
    # Clip to [0, 1]
    f = np.clip(f, 0, 1)
    # Project to simplex with sum(f) <= 1
    if f.sum() <= 1:
        return f
    u = np.sort(f)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - 1) / (np.arange(len(f)) + 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(f - theta, 0)
    return w

def kelly_optimizer(
    tickers: List[str],
    start: str = "2020-01-01",
    end: str = "2023-08-01",
    lr: float = 0.1,
    max_iter: int = 1000,
    tol: float = 1e-6,
    verbose: bool = True
) -> Dict[str, float]:
    try:
        R, names = get_log_returns(tickers, start, end)
        n_assets = R.shape[1]
        f = np.ones(n_assets) / n_assets

        for _ in range(max_iter):
            grad = kelly_gradient(f, R)
            if np.any(np.isnan(grad)):
                raise ValueError("Gradient exploded or log domain violated.")

            f_new = f - lr * grad
            f_new = project_to_constraints(f_new)

            if np.linalg.norm(f_new - f) < tol:
                break
            f = f_new

        port_ret = R @ f
        result = dict(zip(names, np.round(f, 6)))

        if verbose:
            print("Kelly Portfolio Weights:")
            for name, weight in result.items():
                print(f"  {name}: {weight:.4f}")
            print("\nPortfolio Stats:")
            print(f"  Mean daily return   : {np.mean(port_ret):.6f}")
            print(f"  Std deviation       : {np.std(port_ret):.6f}")
            print(f"  Log-growth rate     : {np.mean(np.log(1 + port_ret)):.6f}")
            print(f"  Geometric growth    : {np.exp(np.mean(np.log(1 + port_ret))) - 1:.6f} per day")

        return result

    except Exception as e:
        print(f"Error during optimization: {e}")
        return {}

if __name__ == "__main__":
    tickers = ["MSTR", "MSFL", "DDOG", "NVDX", "JEPQ"]
    weights = kelly_optimizer(tickers, start="2025-01-01", end="2025-08-01")
