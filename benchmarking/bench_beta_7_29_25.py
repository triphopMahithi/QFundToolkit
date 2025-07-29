import yfinance as yf
import numpy as np
import logging
import time
import gc


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

def calculate_beta_covariance(
    target_symbol: str,
    index_symbol: str,
    period: str = "1y",
    auto_adjust: bool = True,
    round_digits: int = 6
) -> float:
    """
    Calculate the beta of a target stock relative to a market index using log returns.

    Args:
        target_symbol (str): The stock ticker of the target asset.
        index_symbol (str): The stock ticker of the market index (e.g., "^GSPC").
        period (str, optional): Time period to fetch data (e.g., '1mo', '1y', '5y'). Defaults to '1y'.
        auto_adjust (bool, optional): Whether to use adjusted prices for dividends/splits. Defaults to True.
        round_digits (int, optional): Number of decimal places to round the result. Defaults to 6.

    Returns:
        float: The beta coefficient, rounded to specified digits.

    :example:
            >> beta = beta_s("SCC.BK", "^SET.BK", period="1y", auto_adjust=True)
            >> print(f"Beta: {beta}")
    
    output:
            >> INFO:__main__:Fetching data for SCC.BK and ^SET.BK over period '1y'...
            >> INFO:__main__:Calculated beta: 1.38398
            >> Beta: 1.38398

    Raises:
        ValueError: If data cannot be retrieved or aligned correctly.
    """
    #logger.info(f"Fetching data for {target_symbol} and {index_symbol} over period '{period}'...")

    try:
        target_df = yf.download(target_symbol, period=period, progress=False, auto_adjust=auto_adjust)
        index_df = yf.download(index_symbol, period=period, progress=False, auto_adjust=auto_adjust)
    except Exception as e:
        logger.error("Failed to fetch data from Yahoo Finance.")
        raise e

    if target_df.empty or index_df.empty:
        raise ValueError("One or both datasets could not be fetched or are empty.")

    # Compute log returns
    target_returns = np.log(target_df["Close"] / target_df["Close"].shift(1)).dropna().squeeze()
    index_returns = np.log(index_df["Close"] / index_df["Close"].shift(1)).dropna().squeeze()

    # Align by date index
    target_returns, index_returns = target_returns.align(index_returns, join="inner")

    if len(target_returns) < 2:
        raise ValueError("Insufficient aligned data points after cleaning.")

    # Calculate beta
    covariance = np.cov(target_returns, index_returns)[0][1]
    variance = np.var(index_returns)
    beta = covariance / variance

    #logger.info(f"Calculated beta: {round(beta, round_digits)}")

    return round(beta, round_digits)




# Execution Time (Wall Time)
print("HEAD: Execution Time (Wall Time)")
start_old = time.perf_counter()
beta1 = calculate_beta_covariance("AAPL", "^GSPC", period="1y")
end_old = time.perf_counter()
delta_old = end_old - start_old
print(f"beta_s (classic): {beta1}, Time: {delta_old:.6f} sec")

start_new = time.perf_counter()
beta2 = calculate_beta_lstsq("AAPL", "^GSPC", period="1y")
end_new = time.perf_counter()
delta_new = end_new - start_new
print(f"beta_s (new): {beta2}, Time: {delta_new:.6f} sec")

# Accuracy and Correctness Evaluation
print("=== Accuracy and Correctness ===")
print(f"Absolute Difference (|β1 - β2|): {abs(beta1 - beta2):.8f}")
print("--- Relative Ratios ---")
print(f"Beta Ratio (β2 / β1): {beta2 / beta1:.8f}")
print(f"Execution Time Ratio (T2 / T1): {delta_new / delta_old:.8f}")
