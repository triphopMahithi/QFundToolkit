import yfinance as yf
import numpy as np
import logging
from memory_profiler import memory_usage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

# Memory Usage
def wrapper():
    return calculate_beta_covariance("AAPL", "^GSPC", period="1y")

if __name__ == "__main__":
    MAX_ITERATIONS = 10000
    mem_usage_old = memory_usage(wrapper, max_iterations=MAX_ITERATIONS)
    print(f"itereration : {MAX_ITERATIONS}")
    print(f"Memory used (old): {max(mem_usage_old) - min(mem_usage_old):.4f} MB")

