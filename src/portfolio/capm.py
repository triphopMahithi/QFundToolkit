import os
import logging
import pandas as pd
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def capm(
    risk_free_rate: float,
    beta: float,
    market_return: Optional[float] = None,
    country: str = 'United States',
    type: str = 'stan'
) -> float:
    
    """
    Calculate the expected return of an asset using the CAPM (Capital Asset Pricing Model) formula.

    If type is "stan" or "standard", the function will attempt to automatically read the market return
    (Equity Risk Premium) for the specified country from the most recent country risk data CSV in the 'log/' folder.
    If type is not standard, it will perform inverse CAPM using the provided market_return value.

    Args:
        risk_free_rate (float): The risk-free rate of return (e.g., 0.02 for 2%).
        beta (float): The beta of the asset.
        market_return (Optional[float], optional): The expected market return or ERP. 
            Required only if type is not 'standard'. Defaults to None.
        country (str, optional): The name of the country to lookup ERP from CSV. 
            Used only when type is 'stan' or 'standard'. Defaults to 'United States'.
        type (str, optional): Calculation mode. 'stan' or 'standard' uses auto ERP lookup. 
            Other values use inverse CAPM with manual market_return. Defaults to 'stan'.

    Returns:
        float: The expected return computed using the CAPM formula.

    Raises:
        FileNotFoundError: If the log folder or data file is not found.
        ValueError: If country is not found or market_return is missing when required.

    Examples:
        >>> # Example 1: Standard CAPM using auto ERP from latest file
        >>> cap_m(risk_free_rate=0.02, beta=1.2, country="Thailand", type="standard")
        0.218

        >>> # Example 2: Manual market return (inverse CAPM)
        >>> cap_m(risk_free_rate=0.02, beta=1.2, market_return=0.08, type="custom")
        -0.052
    """

    if type.lower() not in ['stan', 'standard']:
        if market_return is None:
            raise ValueError("Market return must be provided for non-standard CAPM.")
        return risk_free_rate - beta * (market_return - risk_free_rate)

    log_dir = "log"
    if not os.path.exists(log_dir):
        raise FileNotFoundError("Log directory does not exist.")

    csv_files = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.startswith("country_risk_data_") and f.endswith(".csv")
    ]
    if not csv_files:
        raise FileNotFoundError("No country_risk_data_*.csv found in log directory.")

    def extract_date(file_name: str) -> datetime:
        try:
            date_str = file_name.split("country_risk_data_")[-1].split(".csv")[0]
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return datetime.min

    latest_file = max(csv_files, key=lambda f: extract_date(os.path.basename(f)))
    logger.info(f"Using data from file: {latest_file}")

    df = pd.read_csv(latest_file)
    row = df[df["Country"].str.lower() == country.lower()]
    if row.empty:
        raise ValueError(f"Country '{country}' not found in {latest_file}")

    erp = row.iloc[0]["Equity Risk Premium"]
    if pd.isnull(erp):
        raise ValueError(f"Equity Risk Premium is missing for country '{country}'.")

    expected_return = risk_free_rate + beta * (erp - risk_free_rate)
    logger.info(f"Computed CAPM for {country}: {expected_return:.6f}")
    return expected_return

