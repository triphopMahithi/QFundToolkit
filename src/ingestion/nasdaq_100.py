"""Yahoo Finance NASDAQ-100 Data Downloader.

This script fetches historical stock data for NASDAQ-100 symbols using
Yahoo Finance API (yfinance), supports batch downloads, and saves data
to Pickle and CSV files for later use.

Usage:
    python main_production.py

Example:
    >>> from main_production import fetch_multi_stock_data
    >>> symbols = get_nasdaq_100_list()
    >>> df = fetch_multi_stock_data(symbols)
    >>> print(df.head())

Environment Variables:
    MONGO_URI       : MongoDB connection URI (optional)
    DB_NAME         : MongoDB database name
    COLLECTION_NAME : MongoDB collection name
    PERIOD          : Data period (e.g., "1y")
    INTERVAL        : Data interval (e.g., "1d")
    BATCH_SIZE      : Number of tickers per request batch
    REQUEST_DELAY   : Delay (seconds) between batches
    CACHE_DIR       : Directory for cached files
    NASDAQ          : URL for NASDAQ-100 tickers list

"""

import os
import math
import time
import logging
from typing import List
import requests
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from utils import save_to_pickle, save_to_csv  # Optional: save_to_mongo


# ==============================
# CONFIGURATION
# ==============================
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
PERIOD = os.getenv("PERIOD", "1y")
INTERVAL = os.getenv("INTERVAL", "1d")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 25))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", 2))
CACHE_DIR = os.getenv("CACHE_DIR", "cache")
NASDAQ_URL = os.getenv("NASDAQ")

logger = logging.getLogger("YFMultiDownloader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ==============================
# FUNCTIONS
# ==============================

def safe_download(symbols: List[str], period: str, interval: str, batch_size: int = 25) -> pd.DataFrame:
    """Try to download all symbols at once, fallback to batch mode if failed."""
    try:
        logger.info("Attempting single-call download for all tickers...")
        df = yf.download(" ".join(symbols), period=period, interval=interval, group_by='ticker', threads=True, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            logger.info("Single-call download successful.")
            return df
        else:
            logger.warning("Single-call returned empty DataFrame, switching to batch mode.")
    except Exception as e:
        logger.error(f"Single-call download failed: {e}. Switching to batch mode...")

    # --- Fallback: Batch mode ---
    all_data = []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        logger.info(f"Downloading batch {i//batch_size+1}: {batch}")
        try:
            data = yf.download(" ".join(batch), period=period, interval=interval, group_by='ticker', threads=True, progress=False)
            if isinstance(data, pd.DataFrame) and not data.empty:
                all_data.append(data)
            else:
                logger.warning(f"No data for batch {batch}")
        except Exception as e:
            logger.error(f"Failed batch {batch}: {e}")
        time.sleep(1)  # avoid rate-limit

    if not all_data:
        logger.error("No data retrieved for any symbol.")
        return pd.DataFrame()

    return pd.concat(all_data, axis=1)

    
def get_nasdaq_100_list() -> List[str]:
    """Fetch NASDAQ-100 ticker symbols from the specified URL.

    Returns:
        List[str]: A list of ticker symbols.
    """
    if not NASDAQ_URL:
        raise ValueError("Environment variable 'NASDAQ' is not set.")

    try:
        resp = requests.get(NASDAQ_URL, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch NASDAQ-100 list: {e}")
        return []

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, 'html.parser')
    tickers = [
        row.select("td")[1].text.strip()
        for row in soup.select("table tbody tr")
        if row.select("td") and row.select("td")[0].text.strip().isdigit()
        and int(row.select("td")[0].text.strip()) <= 100
    ]

    logger.info(f"Retrieved {len(tickers)} NASDAQ-100 tickers.")
    return tickers


def fetch_multi_stock_data(symbols: List[str]) -> pd.DataFrame:
    """Fetch historical stock data for a list of symbols in batches.

    Args:
        symbols (List[str]): List of ticker symbols.

    Returns:
        pd.DataFrame: Combined MultiIndex DataFrame of stock data.
    """
    if not symbols:
        raise ValueError("Symbol list is empty.")

    all_data = []
    total_batches = math.ceil(len(symbols) / BATCH_SIZE)

    for i in range(total_batches):
        batch = symbols[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        logger.info(f"Fetching batch {i+1}/{total_batches}: {batch}")

        try:
            data = yf.download(
                tickers=batch,
                period=PERIOD,
                interval=INTERVAL,
                group_by='ticker',
                auto_adjust=False,
                threads=True,
                progress=False
            )
            if isinstance(data, pd.DataFrame) and not data.empty:
                all_data.append(data)
            else:
                logger.warning(f"No data retrieved for batch {i+1}.")
        except Exception as e:
            logger.error(f"Error fetching batch {i+1}: {e}")

        time.sleep(REQUEST_DELAY)

    if not all_data:
        logger.error("No data was downloaded for any ticker.")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, axis=1)
    logger.info(f"Successfully fetched data for {len(symbols)} symbols.")
    return combined_df


# TODO: INCREMENTAL FUNCTION

# ==============================
# MAIN PROCESS
# ==============================
def main():
    symbols = get_nasdaq_100_list()
    if not symbols:
        logger.error("No symbols found. Exiting.")
        return

    df = safe_download(symbols, PERIOD, INTERVAL, batch_size=25)
    if df.empty:
        logger.warning("No data to save. Exiting.")
        return

    save_to_pickle(df, PERIOD, INTERVAL, CACHE_DIR)
    save_to_csv(df, PERIOD, INTERVAL, CACHE_DIR)
    logger.info("Completed fetching and saving NASDAQ-100 stock data.")

if __name__ == "__main__":
    main()
