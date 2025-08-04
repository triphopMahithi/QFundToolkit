import yfinance as yf
import numpy as np
import logging
import gc
# Set up default logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import yfinance as yf
import numpy as np
import gc
import logging

logger = logging.getLogger(__name__)

def beta_s(
    target_symbol: str,
    index_symbol: str,
    period: str = "1y",
    auto_adjust: bool = True,
    round_digits: int = 6,
    fit_intercept: bool = True
) -> float:
    """
    Estimate beta of a target stock relative to a market index using least squares (linear algebra).

    Args:
        target_symbol (str): Ticker of the target stock (e.g., "AAPL").
        index_symbol (str): Ticker of the market index (e.g., "^GSPC").
        period (str, optional): Period to fetch data (e.g., "1y", "6mo"). Defaults to "1y".
        auto_adjust (bool, optional): Whether to use adjusted prices. Defaults to True.
        round_digits (int, optional): Decimal places to round the result. Defaults to 6.
        fit_intercept (bool, optional): Whether to fit an intercept (alpha). Defaults to True.

    Returns:
        float: Beta value rounded to desired decimal places.

    Example:
            >>> beta1 = beta_s("AAPL", "^GSPC", fit_intercept=True)
            >>> beta2 = beta_s("AAPL", "^GSPC", fit_intercept=False)
            INFO:__main__:Downloading data for AAPL and ^GSPC...
            INFO:__main__:Calculated beta: 1.221255 (intercept fitted: True)
            INFO:__main__:Downloading data for AAPL and ^GSPC...
            INFO:__main__:Calculated beta: 1.217822 (intercept fitted: False)
    Raises:
        ValueError: If data fetching or alignment fails.
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

    # Design matrix
    if fit_intercept:
        X_design = np.column_stack((np.ones(X.shape[0]), X))  # with intercept
    else:
        X_design = X  # no intercept

    # Least squares estimation
    coeffs = np.linalg.lstsq(X_design, Y, rcond=None)[0]
    beta = coeffs[-1]  # the slope (last element, with or without intercept)

    # Cleanup
    del X, Y, X_design
    gc.collect()

    logger.info(f"Calculated beta: {round(beta, round_digits)} (intercept fitted: {fit_intercept})")
    return round(beta, round_digits)

