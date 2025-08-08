import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict
from scipy.optimize import minimize_scalar

def kelly_criterion(b: Union[float, np.ndarray, pd.Series],
                    p: Union[float, np.ndarray, pd.Series],
                    l: Union[float, np.ndarray, pd.Series] = 1,
                    include_principal: bool = True,
                    round_digits: int = 6) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate the optimal fraction of capital to invest using the generalized Kelly Criterion.
    
    Formula:
        If include_principal = True:
            R_win = 1 + bf
            R_loss = 1 - lf
        Then:
            f* = (bp - ql) / (bq + pl)

        If include_principal = False:
            R_win = bf
            R_loss = -lf
        Then:
            f* = (bp - ql) / (bp + ql)

    Args:
        b (float | array-like): Gain multiplier (e.g., 0.3 means +30% gain).
        p (float | array-like): Probability of gain.
        l (float | array-like): Loss multiplier (e.g., 0.2 means -20% loss).
        round_digits (int): Decimal places to round the result.
        include_principal (bool): Whether to include principal in log return calculation.

    Returns:
        float | np.ndarray | pd.Series: Optimal investment fraction(s).

    Raises:
        ValueError: If inputs contain NaNs or incompatible shapes.

    Examples:
        >>> kelly_criterion(2, 0.6, include_principal=False)
        0.4

        >>> kelly_criterion([2, 3], [0.6, 0.7], l=0.5, include_principal=False)
        [1.  1.3]

    """
    b = np.asarray(b, dtype=float)
    p = np.asarray(p, dtype=float)
    l = np.asarray(l, dtype=float)

    try:
        b, p, l = np.broadcast_arrays(b, p, l)
    except ValueError as e:
        raise ValueError(f"Incompatible shapes: {e}")

    if np.any(np.isnan(b)) or np.any(np.isnan(p)) or np.any(np.isnan(l)):
        raise ValueError("Inputs contain NaN values.")

    q = 1 - p

    if include_principal:
        # Use full Kelly formula (includes principal)
        numerator = b * p - q * l
        denominator = b * q + p * l
    else:
        # Use gain/loss only version (excludes principal)
        numerator = b * p - q * l
        denominator = b * l

    f_star = numerator / denominator
    result = np.round(f_star, round_digits)

    if isinstance(b, pd.Series):
        return pd.Series(result, index=b.index)
    elif result.ndim == 0:
        return float(result)
    else:
        return result

def empirical_kelly(
    returns: Union[np.ndarray, list],
    f_bounds: Tuple[float, float] = (0.0, 2.0),
    resolution: float = 1e-4,
    round_digits: int = 6,
    risk_free_rate: float = 0.0,
    subtract_mean: bool = False
) -> float:
    """
    Estimate the optimal Kelly fraction from historical returns using empirical optimization.

    This function computes the Kelly fraction f* that maximizes the expected logarithmic
    growth of wealth given a series of historical returns.

    Args:
        returns (Union[np.ndarray, list]):
            A list or numpy array of historical returns (e.g. daily or monthly), in decimal form.
            For example, 1% return should be input as 0.01.
        f_bounds (Tuple[float, float], optional):
            Lower and upper bounds for the optimization of Kelly fraction. Default is (0.0, 2.0).
        resolution (float, optional):
            Precision of the optimizer (`xatol` in `minimize_scalar`). Smaller = higher precision.
            Default is 1e-4.
        round_digits (int, optional):
            Number of decimal places to round the result. Default is 6.
        risk_free_rate (float, optional):
            Risk-free rate to subtract from all returns, in same scale. Default is 0.0.
        subtract_mean (bool, optional):
            If True, subtract mean from return series (useful for excess return calculations).

    Returns:
        float:
            The optimal Kelly fraction f* (rounded), or 0.0 if optimization fails.

    Raises:
        ValueError: If input returns are not valid.

    Example:
        >>> returns = [0.01, -0.02, 0.015, -0.01, 0.005]
        >>> empirical_kelly(returns)
        0.000048
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.ndim != 1:
        raise ValueError("Input returns must be a 1D array or list.")

    if len(r) < 2:
        return 0.0

    if subtract_mean:
        r = r - np.mean(r)

    r = r - risk_free_rate
    def objective(f: float) -> float:
        """Negative expected log return (to be minimized)."""
        if np.any(1 + f * r <= 0):
            return np.inf  # Invalid domain for log
        return -np.mean(np.log(1 + f * r))

    result = minimize_scalar(
        objective,
        bounds=f_bounds,
        method='bounded',
        options={'xatol': resolution}
    )
    if not result.success or np.isnan(result.x):
        return 0.0

    return round(result.x, round_digits)

def allocate_budget(weights: Dict[str, float], total_budget: float) -> pd.DataFrame:
    """
    Allocate a given total budget based on normalized portfolio weights.

    Args:
        weights (Dict[str, float]): A dictionary mapping asset symbols to their normalized weights.
                                    All weights should sum to approximately 1.0 (or 100%).
        total_budget (float): The total amount of money to allocate (e.g., in THB).

    Returns:
        pd.DataFrame: A DataFrame containing columns:
            - 'Symbol': Asset symbol
            - 'Weight (%)': Portfolio weight as a percentage
            - 'Allocated (THB)': Budget amount allocated to that asset

    Example:
        >>> weights = {'AAPL': 0.25, 'MSFT': 0.35, 'TSLA': 0.4}
        >>> allocate_budget(weights, total_budget=100000)
           Symbol  Weight (%)  Allocated (THB)
        0   TSLA        40.0         40000.00
        1   MSFT        35.0         35000.00
        2   AAPL        25.0         25000.00
    """
    if not weights or total_budget <= 0:
        raise ValueError("Weights must be non-empty and total_budget must be positive.")

    if not isinstance(weights, dict):
        raise TypeError("weights must be a dictionary of {symbol: weight}.")

    if abs(sum(weights.values()) - 1.0) > 1e-3:
        raise ValueError("Weights must be normalized to sum to 1.0.")

    allocation = {symbol: weight * total_budget for symbol, weight in weights.items()}

    df = pd.DataFrame({
        'Symbol': list(allocation.keys()),
        'Weight (%)': [round(weight * 100, 2) for weight in weights.values()],
        'Allocated (THB)': [round(amount, 2) for amount in allocation.values()]
    })

    return df.sort_values(by='Weight (%)', ascending=False).reset_index(drop=True)

