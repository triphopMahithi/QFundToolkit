import pandas as pd
import numpy as np
import yfinance as yf
from typing import Union

import numpy as np
import pandas as pd
from typing import Union

def mean_return(asset_returns: Union[np.ndarray, pd.Series, pd.DataFrame],
                weights: Union[np.ndarray, list, None] = None,
                round_digits: int = 6) -> float:
    """Calculate the mean return of a portfolio.

    This function computes the mean return of a portfolio based on individual
    asset returns and their corresponding weights. If weights are not provided,
    the function assumes equal weighting for all assets.

    Args:
        asset_returns (Union[np.ndarray, pd.Series, pd.DataFrame]):
            A 1D or 2D array-like structure containing returns of assets.
            - If 1D: treated as pre-computed portfolio returns.
            - If 2D: each column represents an asset, and each row represents a time period.
        weights (Union[np.ndarray, list, None], optional):
            Portfolio weights for each asset. The weights will be normalized to sum to 1.
            Defaults to None (equal weights).
        round_digits (int, optional):
            Number of decimal places to round the result. Defaults to 6.

    Returns:
        float: Mean return of the portfolio over the given period.

    Raises:
        ValueError: If weights are provided but do not match the number of assets.

    Examples:
        >>> import numpy as np
        >>> returns = np.array([[0.01, 0.02], [0.03, -0.01], [0.02, 0.01]])
        >>> weights = [0.6, 0.4]
        >>> mean_return(returns, weights)
        0.013667

        >>> # Case with single pre-computed portfolio returns
        >>> portfolio_returns = np.array([0.01, 0.02, -0.005, 0.03])
        >>> mean_return(portfolio_returns)
        0.01375
    """
    returns = np.asarray(asset_returns, dtype=float)

    if returns.ndim == 1:
        return round(float(np.mean(returns)), round_digits)

    n_assets = returns.shape[1]

    if weights is None:
        weights = np.ones(n_assets) / n_assets
    else:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != n_assets:
            raise ValueError(f"Length of weights ({len(weights)}) does not match number of assets ({n_assets}).")
        total_weight = np.sum(weights)
        if total_weight == 0:
            raise ValueError("Sum of weights cannot be zero.")
        weights = weights / total_weight

    # normalize weights 
    total_weight = np.sum(weights)
    if total_weight == 0:
        raise ValueError("Sum of weights cannot be zero.")
    weights = np.asarray(weights, dtype=float) / total_weight

    portfolio_returns = np.dot(returns, weights)

    weighted_average_return = np.mean(portfolio_returns)

    return round(float(weighted_average_return), round_digits)

def volatility(portfolio_returns: Union[np.ndarray, pd.Series],
               mean_ret: Union[float, np.ndarray, pd.Series, None] = None,
               round_digits: int = 6) -> float:  
    """Compute the standard deviation (volatility) of portfolio returns.

    Uses population standard deviation (N denominator, not N-1) unless specified otherwise.

    Args:
        portfolio_returns (Union[np.ndarray, pd.Series]):
            Array or Series of portfolio returns.
        mean_ret (Union[float, np.ndarray, pd.Series, None], optional):
            Pre-computed mean return. If not provided, will be calculated internally.
        round_digits (int, optional):
            Number of decimal places to round the result. Defaults to 6.

    Returns:
        float: Volatility of the returns (standard deviation).

    Raises:
        ValueError: If input data contains NaN or mismatched shapes.

    Examples:
        >>> r = np.array([0.01, 0.02, -0.005, 0.015])
        >>> volatility(r)
        0.009354
    """

    returns = np.asarray(portfolio_returns)

    if returns.ndim != 1:
        raise ValueError("portfolio_returns must be 1D array or Series")
    
    if np.isnan(returns).any():
        raise ValueError("portfolio_returns contains NaN values")
    
    if mean_ret is None:
        mean_ret = np.mean(returns)
    else:
        mean_ret = float(mean_ret)
        if not np.isscalar(mean_ret):
            raise ValueError("mean_ret must be a scalar value")
        
    # vectorized 
    variance = np.mean((returns - mean_ret) ** 2)
    volatility = np.sqrt(variance)

    return round(float(volatility), round_digits)

def maxDD(portfolio_values: Union[np.ndarray, pd.Series],
          return_series: bool = False,
          round_digits: int = 6) -> Union[float, pd.Series]:
    
    """Calculate the Maximum Drawdown (MDD) of a portfolio over time.

    Args:
        portfolio_values (Union[np.ndarray, pd.Series]):
            Portfolio values or cumulative returns over time (1D).
            Can be a NumPy array or Pandas Series with datetime index.
        return_series (bool, optional):
            If True, returns the full drawdown series instead of only MDD value.
        round_digits (int, optional):
            Number of decimal places to round the result.

    Returns:
        Union[float, pd.Series]:
            - If return_series=False: maximum drawdown (as a negative float).
            - If return_series=True: full drawdown series (same index as input).

    Raises:
        ValueError: If input is not 1D or contains NaN.

    Example:
        >>> import yfinance as yf
        >>> data = yf.download("NVDA", period="1y")['Close']
        >>> mdd = maxDD(data)
        >>> dd_series = maxDD(data, return_series=True)
    """
    
    values = np.asarray(portfolio_values, dtype=float)

    if values.ndim != 1:
        raise ValueError("portfolio_values must be a 1D array or Series.")

    if np.isnan(values).any():
        raise ValueError("portfolio_values contains NaN values.")

    V_peak = np.maximum.accumulate(values)
    drawdowns = (values - V_peak) / V_peak

    if isinstance(portfolio_values, pd.Series):
        drawdowns = pd.Series(drawdowns, index=portfolio_values.index)

    if return_series:
        return drawdowns
    else:
        return round(float(np.min(drawdowns)), round_digits)

def VaR(returns: Union[np.ndarray, pd.Series],
                  confidence_level: float = 0.95,
                  method: str = "historical",
                  round_digits: int = 6) -> float:
    """
    Calculate the Value at Risk (VaR) for a given returns series.

    This function computes the potential loss in a portfolio (or single asset)
    over a specified confidence level using either Historical Simulation or
    Parametric (Variance-Covariance) approach.

    Args:
        returns (Union[np.ndarray, pd.Series]):
            Array or Series of returns (percentage changes, e.g., 0.01 = 1%).
        confidence_level (float, optional):
            Confidence level for VaR calculation (default = 0.95).
        method (str, optional):
            Calculation method: 'historical' or 'parametric'.
        round_digits (int, optional):
            Number of decimal places to round the result.

    Returns:
        float: The Value at Risk (VaR) as a negative float, representing
               the potential loss at the given confidence level.

    Raises:
        ValueError: If inputs are invalid or method is unsupported.

    Example:
        >>> import yfinance as yf
        >>> data = yf.download("DDOG", period="1y")['Adj Close']
        >>> returns = data.pct_change().dropna()
        >>> calculate_var(returns, 0.95, 'historical')
        -0.0521
    """
    arr = np.asarray(returns, dtype=float)

    if arr.ndim != 1:
        raise ValueError("Input returns must be a 1D array or Series.")
    if np.isnan(arr).any():
        raise ValueError("Input returns contain NaN values.")
    if not (0 < confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1.")

    alpha = (1 - confidence_level) * 100  # quantile percentage

    if method.lower() == "historical":
        var = np.percentile(arr, alpha)

    elif method.lower() == "parametric":
        mean = np.mean(arr)
        std_dev = np.std(arr, ddof=1)
        z_score = abs(np.percentile(np.random.normal(0, 1, 1000000), alpha))
        var = mean - z_score * std_dev
    else:
        raise ValueError("Unsupported method. Use 'historical' or 'parametric'.")

    return round(float(var), round_digits)


def tracking_error(portfolio_returns: Union[np.ndarray, pd.Series],
                   benchmark_returns: Union[np.ndarray, pd.Series, None] = None,
                   index: str = "^GSPC",
                   period: str = "1y",
                   round_digits: int = 6) -> float:
    """
    Calculate the Tracking Error (TE) between portfolio and benchmark returns.

    Args:
        portfolio_returns (Union[np.ndarray, pd.Series]):
            Daily returns of the portfolio (1D array or Series).
        benchmark_returns (Union[np.ndarray, pd.Series, None], optional):
            Daily returns of the benchmark. If None, data will be downloaded from yfinance.
        index (str, optional):
            Benchmark ticker symbol (default: "^GSPC").
        period (str, optional):
            Period to fetch benchmark data if benchmark_returns is None (default: "1y").
        round_digits (int, optional):
            Decimal places to round the result.

    Returns:
        float: Tracking Error as a positive float.

    Example:
        >>> data = yf.download("DDOG", period="1y")['Close'].squeeze()
        >>> port_ret = data.pct_change().dropna()
        >>> te = tracking_error(port_ret)
        0.021744
    """
    port = pd.Series(portfolio_returns).dropna()
    
    if benchmark_returns is None:
        bench_prices = yf.download(tickers=index, period=period)['Close'].squeeze()
        bench_ret = pd.Series(bench_prices).pct_change().dropna()
    else:
        bench_ret = pd.Series(benchmark_returns).dropna()

    df = pd.concat([port, bench_ret], axis=1, join='inner')
    df.columns = ['portfolio', 'benchmark']

    if df.empty:
        raise ValueError("No overlapping data between portfolio and benchmark.")

    active_ret = df['portfolio'] - df['benchmark']

    te = np.std(active_ret, ddof=1)  # sample std
    return round(float(te), round_digits)

