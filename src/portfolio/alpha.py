import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Union, Dict

def jensen_alpha(portfolio_return: Union[float, np.ndarray],
                 risk_free_rate: Union[float, np.ndarray],
                 beta: Union[float, np.ndarray],
                 market_return: Union[float, np.ndarray],
                 round_digits : int = 6) -> Union[float, np.ndarray]:
    
    """Calculate Jensen's Alpha for portfolio performance.

    Supports both single values and NumPy arrays for batch calculation.

    Args:
        portfolio_return: Portfolio return(s), decimal or array-like.
        risk_free_rate: Risk-free rate(s), decimal or array-like.
        beta: Portfolio beta(s), decimal or array-like.
        market_return: Market return(s), decimal or array-like.

    Returns:
        Jensen's Alpha value(s) as float or NumPy array.

    Examples:
        >>> jensen_alpha(0.08, 0.02, 1.1, 0.07)
        0.005

        >>> r_p_array = [0.08, 0.12, 0.05]
        >>> r_f_array = 0.02
        >>> beta_array = [1.1, 0.9, 0.7]
        >>> r_m_array = [0.07, 0.10, 0.04]
        >>> jensen_alpha(r_p_array, r_f_array, beta_array, r_m_array)
        array([0.005, 0.028, 0.016])

    Raises:
        ValueError: If array inputs have inconsistent lengths.
    """

    portfolio_return = np.asarray(portfolio_return)
    risk_free_rate = np.asarray(risk_free_rate)
    beta = np.asarray(beta)
    market_return = np.asarray(market_return)

    # Validate shapes
    try:
        np.broadcast_shapes(
            np.shape(portfolio_return),
            np.shape(risk_free_rate),
            np.shape(beta),
            np.shape(market_return)
        )
    except ValueError:
        raise ValueError("Input arrays must have the same length or be scalar values.")

    alpha = portfolio_return - (risk_free_rate + beta*(market_return - risk_free_rate))
    return np.round(alpha, round_digits)


def multi_factor(portfolio_returns: Union[np.ndarray, pd.Series],
                 risk_free_rate: Union[np.ndarray, pd.Series],
                 market_returns: Union[np.ndarray, pd.Series],
                 SMB: Union[np.ndarray, pd.Series],
                 HML: Union[np.ndarray, pd.Series],
                 MOM: Union[np.ndarray, pd.Series] = None,
                 model: str = "carhart",
                 round_digits: int = 6) -> Dict[str, float]:
    """Estimate multi-factor regression alpha and betas using Fama-French (3-Factor)
    or Carhart (4-Factor) model.

    The function uses OLS regression to estimate factor loadings:

    Fama-French 3-Factor:
        R_p - R_f = α + β_m(MKT - R_f) + β_s(SMB) + β_h(HML) + ε

    Carhart 4-Factor:
        R_p - R_f = α + β_m(MKT - R_f) + β_s(SMB) + β_h(HML) + β_mom(MOM) + ε

    Args:
        portfolio_returns (array-like): Portfolio returns (decimal values).
        risk_free_rate (array-like): Risk-free rate (decimal values).
        market_returns (array-like): Market returns (decimal values).
        SMB (array-like): Small Minus Big factor returns.
        HML (array-like): High Minus Low factor returns.
        MOM (array-like, optional): Momentum factor returns, required for Carhart model.
        model (str, optional): Choose regression model. Options:
            - "carhart": Carhart 4-Factor (default)
            - "fama": Fama-French 3-Factor
        round_digits (int, optional): Number of decimal places for output. Defaults to 6.

    Returns:
        dict: Dictionary containing regression results:
            - alpha (float)
            - beta_market (float)
            - beta_smb (float)
            - beta_hml (float)
            - beta_mom (float) [only for Carhart]
            - r_squared (float)

    Raises:
        ValueError: If inputs have incompatible shapes or insufficient data points.

    Examples:
        >>> portfolio = [0.05, 0.08, 0.03, 0.07, 0.06, 0.09, 0.04, 0.11, 0.10, 0.07]
        >>> risk_free = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        >>> market = [0.04, 0.07, 0.02, 0.06, 0.05, 0.08, 0.03, 0.09, 0.08, 0.06]
        >>> SMB = [0.02, 0.03, 0.01, 0.025, 0.02, 0.035, 0.015, 0.03, 0.025, 0.02]
        >>> HML = [0.015, 0.02, 0.005, 0.018, 0.015, 0.025, 0.01, 0.02, 0.018, 0.012]
        >>> MOM = [0.01, 0.015, -0.002, 0.012, 0.01, 0.02, 0.005, 0.015, 0.013, 0.009]
        >>> multi_factor(portfolio_returns=portfolio,
        ...           risk_free_rate=risk_free,
        ...           market_returns=market,
        ...           SMB=SMB,
        ...           HML=HML,
        ...           MOM=MOM)
        {'alpha': 0.006512, 'r_squared': 0.995387, 'beta_market': 1.399629, 
        'beta_smb': -1.244527, 'beta_hml': 1.899814, 'beta_mom': -1.410019}


        >>> portfolio = [0.05, 0.08, 0.03, 0.07, 0.06, 0.09, 0.04, 0.11, 0.10, 0.07]
        >>> risk_free = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        >>> market =    [0.04, 0.07, 0.02, 0.06, 0.05, 0.08, 0.03, 0.09, 0.08, 0.06]
        >>> SMB =       [0.02, 0.03, 0.01, 0.025, 0.02, 0.035, 0.015, 0.03, 0.025, 0.02]
        >>> HML =       [0.015, 0.02, 0.005, 0.018, 0.015, 0.025, 0.01, 0.02, 0.018, 0.012]
        >>> result_fama = multi_factor(
         ...       portfolio_returns=portfolio,
         ...       risk_free_rate=risk_free,
         ...       market_returns=market,
         ...       SMB=SMB,
         ...       HML=HML,
         ...       MOM=None,       
         ...       model="fama")
        {'alpha': 0.014324, 'r_squared': 0.992973, 'beta_market': 1.356757,
          'beta_smb': -1.254054, 'beta_hml': 0.594595}

    """
    portfolio_returns = np.asarray(portfolio_returns, dtype=float)
    risk_free_rate = np.asarray(risk_free_rate, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)
    SMB = np.asarray(SMB, dtype=float)
    HML = np.asarray(HML, dtype=float)
    MOM = np.asarray(MOM, dtype=float) if MOM is not None else None

    model = model.lower()
    if model not in ["carhart", "fama"]:
        raise ValueError("Model must be either 'carhart' or 'fama'.")

    # Validate shapes
    try:
        if model == "fama":
            np.broadcast_shapes(portfolio_returns.shape, risk_free_rate.shape,
                                market_returns.shape, SMB.shape, HML.shape)
        else:
            if MOM is None:
                raise ValueError("MOM factor is required for Carhart model.")
            np.broadcast_shapes(portfolio_returns.shape, risk_free_rate.shape,
                                market_returns.shape, SMB.shape, HML.shape, MOM.shape)
    except ValueError as e:
        raise ValueError("Input arrays must have compatible shapes.") from e

    # Check data points
    if min(portfolio_returns.size, risk_free_rate.size,
           market_returns.size, SMB.size, HML.size) < 3:
        raise ValueError("Not enough data points for regression.")

    # Compute excess returns
    Y = portfolio_returns - risk_free_rate
    factors = [market_returns - risk_free_rate, SMB, HML]
    factor_names = ["beta_market", "beta_smb", "beta_hml"]

    if model == "carhart":
        factors.append(MOM)
        factor_names.append("beta_mom")

    X = np.column_stack(factors)
    X = sm.add_constant(X)

    try:
        results = sm.OLS(Y, X, missing='drop').fit()
    except Exception as e:
        raise RuntimeError("OLS regression failed.") from e

    params = results.params.round(round_digits)
    output = {
        "alpha": float(params[0]),
        "r_squared": round(float(results.rsquared), round_digits)
    }
    for i, name in enumerate(factor_names, start=1):
        output[name] = float(params[i])

    return output

def regression_alpha(
        portfolio_return: Union[np.ndarray, pd.Series],
        risk_free_rate: Union[np.ndarray, pd.Series],
        market_return: Union[np.ndarray, pd.Series],
        round_digits: int = 6) -> Dict[str, float]:
    """Calculate Regression Alpha using OLS Time-Series Regression.

    This function estimates Jensen's alpha (intercept) and beta (slope) 
    from a time-series regression of excess portfolio returns against 
    excess market returns using the following model:

        R_{p,t} - R_{f,t} = α + β(R_{m,t} - R_{f,t}) + ε_t

    Args:
        portfolio_return (array-like): Portfolio returns (decimal values).
        risk_free_rate (array-like): Risk-free rate values (decimal).
        market_return (array-like): Market returns (decimal values).
        round_digits (int, optional): Number of decimal places to round
            the output. Defaults to 6.

    Returns:
        dict: Dictionary containing:
            - alpha (float): Intercept from regression (Jensen's alpha).
            - beta (float): Slope coefficient.
            - r_squared (float): R² of the regression.

    Raises:
        ValueError: If input arrays have incompatible shapes or
            insufficient data points (<2).

    Examples:
        >>> portfolio = [0.05, 0.08, 0.03, 0.07, 0.06]
        >>> risk_free = [0.01, 0.01, 0.01, 0.01, 0.01]
        >>> market = [0.04, 0.07, 0.02, 0.06, 0.05]
        >>> regression_alpha(portfolio, risk_free, market)
        {'alpha': 0.01, 'beta': 1.0, 'r_squared': 1.0}

        >>> portfolio2 = np.array([0.1, 0.15, 0.12, 0.18, 0.11])
        >>> risk_free2 = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
        >>> market2 = np.array([0.08, 0.12, 0.09, 0.15, 0.10])
        >>> regression_alpha(portfolio2, risk_free2, market2)
        {'alpha': 0.011429, 'beta': 1.142857, 'r_squared': 0.93992}
    """
    portfolio_return = np.asarray(portfolio_return, dtype=float)
    risk_free_rate = np.asarray(risk_free_rate, dtype=float)
    market_return = np.asarray(market_return, dtype=float)

    try:
        np.broadcast_shapes(
            portfolio_return.shape,
            risk_free_rate.shape,
            market_return.shape
        )
    except ValueError as e:
        raise ValueError("Input arrays must have compatible shapes.") from e

    # Check for minimum data points
    if min(portfolio_return.size, risk_free_rate.size, market_return.size) < 2:
        raise ValueError("Not enough data points for regression (min=2).")

    Y = portfolio_return - risk_free_rate
    X = market_return - risk_free_rate
    X = sm.add_constant(X)

    # Fit OLS regression model
    try:
        model = sm.OLS(Y, X, missing='drop')
        results = model.fit()
        output = {
        "alpha": round(float(results.params[0]), round_digits),
        "beta": round(float(results.params[1]), round_digits),
        "r_squared": round(float(results.rsquared), round_digits)
        }
    except Exception as e:
        raise RuntimeError("OLS regression failed.") from e

    return output

def risk_adj(alpha : Union[float, np.ndarray],
             std : Union[float, np.ndarray],
             round_digits : int = 6) -> Union[float, np.ndarray]:
    """Calculate the Appraisal Ratio (Risk-Adjusted Alpha).

    The Appraisal Ratio measures the excess return per unit of unsystematic risk.
    It is calculated as the ratio of Jensen's Alpha to the standard deviation of
    residual (idiosyncratic) risk.

    The formula is:
        Risk-Adjusted Alpha = α / σ

    Args:
        alpha (float or np.ndarray): Jensen's Alpha value(s), decimal or array-like.
        std (float or np.ndarray): Standard deviation of residual risk, decimal or array-like.
        round_digits (int, optional): Number of decimal places to round the result.
            Defaults to 6.

    Returns:
        float or np.ndarray: Risk-adjusted alpha value(s).

    Raises:
        ValueError: If input arrays are incompatible for broadcasting.

    Examples:
        >>> risk_adj(0.05, 0.02)
        2.5

        >>> risk_adj([0.05, 0.08, 0.02], [0.02, 0.04, 0.01])
        array([2.5, 2.0, 2.0])
    """
    alpha = np.asarray(alpha)
    std = np.asarray(std)

    try:
        np.broadcast_shapes(
            np.shape(alpha),
            np.shape(std)
        )
    except ValueError:
        raise ValueError("Input arrays must have the same length or be scalar values.")

    risk_adj = alpha/std
    return np.round(risk_adj, round_digits)

