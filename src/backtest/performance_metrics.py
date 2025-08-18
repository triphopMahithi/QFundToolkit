"""
portfolio_metrics.py

Production-friendly utilities to compare portfolio performance using Yahoo Finance data.

Features
--------
- Clean API with type hints and docstrings
- Robust weight normalization & validation
- Flexible data download window & price field
- Daily returns → annualized metrics (Return, Volatility, Sharpe)
- Safe math (handles zero-vol edge cases)
- Minimal logging hooks (via `logger`)
- Optional CLI
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple
from src.backtest.utils import *
from datetime import date
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise ImportError("yfinance is required. Install with `pip install yfinance`.") from e

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------
# Data structures
# ---------------------------
@dataclass(frozen=True)
class PortfolioMetrics:
    annual_return_pct: float
    annual_volatility_pct: float
    sharpe_ratio: float
    maxdd: float
    calmar: float
    var : float
    cvar : float


    def to_dict(self) -> Dict[str, float]:
        return {
            "Annual Return (%)": self.annual_return_pct,
            "Annual Volatility (%)": self.annual_volatility_pct,
            "Sharpe Ratio": self.sharpe_ratio,
            "MaxDD" : self.maxdd,
            "Calmar Ratio": self.calmar,
            "VaR (%)": self.var,
            "CVaR (%)": self.cvar
        }


# ---------------------------
# Core utilities
# ---------------------------
def normalize_weights(portfolio: Mapping[str, float]) -> Dict[str, float]:
    """Normalize raw weights so they sum to 1.0.

    Accepts positive or negative weights; if all zeros or empty -> ValueError.
    """
    if not portfolio:
        raise ValueError("Portfolio is empty.")
    total = float(sum(portfolio.values()))
    if total == 0:
        raise ValueError("Sum of weights is zero; cannot normalize.")
    normalized = {str(t): float(w) / total for t, w in portfolio.items()}
    # Drop zero-weight tickers to avoid unnecessary columns
    normalized = {t: w for t, w in normalized.items() if w != 0.0}
    return normalized


def union_tickers(*weight_maps: Mapping[str, float]) -> List[str]:
    """Return a sorted unique list of tickers across weight dictionaries."""
    s = set()
    for m in weight_maps:
        s.update([str(k) for k in m.keys()])
    return sorted(s)


def download_prices(
    tickers: Iterable[str],
    start: str | date,
    end: Optional[str | date] = None,
    price_field: str = "Adj Close",
) -> pd.DataFrame:
    """Download OHLCV field (default: Adj Close) as a wide DataFrame.

    Parameters
    ----------
    tickers : Iterable[str]
        Ticker symbols (Yahoo Finance format).
    start, end : str | date
        Date range (inclusive start, exclusive end per yfinance convention).
    price_field : str
        One of the columns provided by yfinance (e.g., 'Adj Close', 'Close').

    Returns
    -------
    pd.DataFrame
        Wide dataframe with Date index and columns=tickers.
    """
    tickers = list(dict.fromkeys([str(t).strip() for t in tickers if str(t).strip()]))
    if not tickers:
        raise ValueError("No tickers provided.")
    logger.debug("Downloading %s field for %d tickers", price_field, len(tickers))
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if price_field not in data.columns.levels[0]:
            raise ValueError(f"Requested price_field '{price_field}' not in downloaded data columns.")
        data = data[price_field]
    else:
        # Single ticker case returns a Series for each field; ensure DataFrame
        data = data.to_frame(name=tickers[0])
    data = data.sort_index().dropna(how="all")
    # Drop columns that are entirely NA
    data = data.dropna(axis=1, how="all")
    if data.empty:
        raise ValueError("Downloaded price data is empty after cleaning.")
    return data


def to_period_returns(
    prices: pd.DataFrame,
    method: str = "pct_change",
    min_periods: int = 1
) -> pd.DataFrame:
    """Compute period-over-period returns from prices.

    method='pct_change' is standard daily simple returns.
    """
    if method != "pct_change":
        raise NotImplementedError("Only method='pct_change' is currently supported.")
    rets = prices.pct_change().dropna(how="all")
    # Optionally enforce min valid observations
    valid_cols = [c for c in rets.columns if rets[c].count() >= min_periods]
    rets = rets[valid_cols]
    if rets.empty:
        raise ValueError("Returns DataFrame is empty after processing.")
    return rets


def portfolio_returns(
    returns: pd.DataFrame,
    weights: Mapping[str, float]
) -> pd.Series:
    """Compute portfolio returns from asset returns and weights.

    Aligns and fills missing tickers with 0 weight.
    """
    weights = dict(weights)
    aligned_weights = np.array([weights.get(t, 0.0) for t in returns.columns], dtype=float)
    port_ret = returns.values @ aligned_weights
    s = pd.Series(port_ret, index=returns.index, name="portfolio_return")
    return s


def compute_metrics(
    port_ret: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
    var_alpha: float = 0.95,
    horizon_days: int = 1
) -> PortfolioMetrics:
    """Compute annualized return, volatility, and Sharpe ratio.

    Parameters
    ----------
    port_ret : pd.Series
        Period returns (e.g., daily simple returns).
    risk_free_rate : float
        Annualized risk-free rate as decimal (e.g., 0.03 for 3%).
    periods_per_year : int
        Trading periods per year (252 for daily).
    """
    if port_ret.empty:
        raise ValueError("Portfolio return series is empty.")
    avg = float(port_ret.mean())
    std = float(port_ret.std(ddof=1))
    ann_ret = avg * periods_per_year
    ann_vol = std * np.sqrt(periods_per_year)

    # add new metric 
    mdd = max_drawdown(port_ret) * 100
    calmar = calmar_ratio(port_ret, risk_free_rate, periods_per_year)
    
    var_1d = historical_var(port_ret, alpha=var_alpha, horizon_days=horizon_days)
    cvar_1d = historical_cvar(port_ret, alpha=var_alpha, horizon_days=horizon_days)
    if ann_vol == 0:
        sharpe = np.nan
    else:
        sharpe = (ann_ret - risk_free_rate) / ann_vol
    return PortfolioMetrics(
        annual_return_pct=ann_ret * 100.0,
        annual_volatility_pct=ann_vol * 100.0,
        sharpe_ratio=sharpe,
        maxdd=mdd,
        calmar=calmar,
        var=var_1d,
        cvar=cvar_1d
    )


def evaluate_portfolios(
    portfolios: Mapping[str, Mapping[str, float]],
    start: str | date,
    end: Optional[str | date] = None,
    price_field: str = "Close",
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252
) -> pd.DataFrame:
    """High-level convenience: download data, compute returns and metrics for many portfolios.

    Parameters
    ----------
    portfolios : dict[name -> weights]
        Each weights dict maps ticker->weight (raw weights ok; normalized internally).
    start, end : date-like
        Download window for prices.
    price_field : str
        'Adj Close' (default) or 'Close', etc.
    risk_free_rate : float
        Annualized risk-free rate.
    periods_per_year : int
        Trading periods per year (e.g., 252 for daily; 52 for weekly, 12 for monthly).

    Returns
    -------
    pd.DataFrame
        Rows are portfolio names, columns: Annual Return (%), Annual Volatility (%), Sharpe Ratio
    """
    # Normalize all portfolios
    normalized = {name: normalize_weights(w) for name, w in portfolios.items()}
    # Union tickers for a single download pass
    all_tickers = union_tickers(*normalized.values())
    prices = download_prices(all_tickers, start=start, end=end, price_field=price_field)
    rets = to_period_returns(prices, method="pct_change")

    out_rows = {}
    for name, weights in normalized.items():
        pr = portfolio_returns(rets, weights)
        metrics = compute_metrics(pr, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year)
        out_rows[name] = metrics.to_dict()
    df = pd.DataFrame.from_dict(out_rows, orient="index")
    return df


# ---------------------------
# CLI (optional)
# ---------------------------
def _build_example() -> Tuple[Dict[str, float], Dict[str, float]]:
    return (
        {"AAPL": 50, "MSFT": 50},
        {"SPY": 60, "TLT": 40},
    )


def main():
    import argparse, sys, json
    parser = argparse.ArgumentParser(description="Compare portfolio metrics using Yahoo Finance data.")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=False, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--rfr", type=float, default=0.03, help="Annual risk-free rate as decimal (default 0.03)")
    parser.add_argument("--price-field", default="Adj Close", help="Price field: 'Adj Close' or 'Close'")
    parser.add_argument("--portfolios", type=str, help="JSON string of {name: {ticker: weight}}. If omitted, uses example.")
    parser.add_argument("--round", type=int, default=2, help="Round output for display (default 2)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    if args.portfolios:
        try:
            portfolios = json.loads(args.portfolios)
            assert isinstance(portfolios, dict)
        except Exception as e:
            print(f"Invalid --portfolios JSON: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        a, b = _build_example()
        portfolios = {"Portfolio A": a, "Portfolio B": b}

    try:
        df = evaluate_portfolios(
            portfolios=portfolios,
            start=args.start,
            end=args.end,
            price_field=args.price_field,
            risk_free_rate=args.rfr,
            periods_per_year=252,
        )
        print(df.round(args.round).to_string())
    except Exception as e:
        logger.exception("Failed to evaluate portfolios.")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
