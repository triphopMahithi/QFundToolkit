import yfinance as yf
import numpy as np
import logging
import gc
from memory_profiler import memory_usage

logger = logging.getLogger(__name__)

def calculate_beta_lstsq(
    target_symbol: str,
    index_symbol: str,
    period: str = "1y",
    auto_adjust: bool = True,
    round_digits: int = 6
) -> float:
    """
    Estimate beta of a target stock relative to a market index using least squares (linear algebra).

    Args:
        target_symbol (str): Ticker of the target stock (e.g., "AAPL").
        index_symbol (str): Ticker of the market index (e.g., "^GSPC").
        period (str, optional): Period to fetch data (e.g., "1y", "6mo"). Defaults to "1y".
        auto_adjust (bool, optional): Whether to use adjusted prices. Defaults to True.
        round_digits (int, optional): Decimal places to round the result. Defaults to 6.

    Returns:
        float: Beta value rounded to desired decimal places.

    Raises:
        ValueError: If data fetching or alignment fails.

    Example:
        >>> beta = beta_s_lstsq("AAPL", "^GSPC", period="1y")
        >>> print(f"Beta: {beta}")
    """
    logger.info(f"Downloading data for {target_symbol} and {index_symbol}...")

    try:
        target_df = yf.download(target_symbol, period=period, progress=False, auto_adjust=auto_adjust)
        index_df = yf.download(index_symbol, period=period, progress=False, auto_adjust=auto_adjust)
    except Exception as e:
        logger.error("Failed to download data from Yahoo Finance.")
        raise e

    if target_df.empty or index_df.empty:
        raise ValueError("One or both datasets are empty.")

    # Compute log returns
    target_returns = np.log(target_df["Close"] / target_df["Close"].shift(1)).dropna().squeeze()
    index_returns = np.log(index_df["Close"] / index_df["Close"].shift(1)).dropna().squeeze()

    # Align by date
    target_returns, index_returns = target_returns.align(index_returns, join="inner")
    if len(target_returns) < 2:
        raise ValueError("Insufficient aligned data points after cleaning.")

    # Convert to numpy arrays
    Y = target_returns.to_numpy().ravel()
    X = index_returns.to_numpy().reshape(-1, 1)
    
    # key: memory usage : hstack -> column_stack
    X_const = np.column_stack((np.ones(X.shape[0]), X))  # Add intercept column

    # Solve using least squares
    coeffs = np.linalg.lstsq(X_const, Y, rcond=None)[0]
    beta = coeffs[1]  # slope coefficient

    # Cleanup
    # key: memory usage : gc
    del X, Y, X_const
    gc.collect()

    logger.info(f"Calculated beta: {round(beta, round_digits)}")
    return round(beta, round_digits)

# Memory Usage
def wrapper():
    return calculate_beta_lstsq("AAPL", "^GSPC", period="1y")

if __name__ == "__main__":
    MAX_ITERATIONS = 10000
    mem_usage_new = memory_usage(wrapper, max_iterations=MAX_ITERATIONS)
    print(f"itereration : {MAX_ITERATIONS}")
    print(f"Memory used (new): {max(mem_usage_new) - min(mem_usage_new):.4f} MB")

