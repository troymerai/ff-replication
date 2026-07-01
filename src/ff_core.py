"""
ff_core.py — Fama-French validation engine (data-source agnostic)

This module is shared by both the US validation and the Korea replication.
As long as the input conventions are met, it works regardless of where the
data comes from:
    - excess_returns : (T x N) DataFrame, excess return of portfolio i (R_i - RF)
    - factors        : (T x K) DataFrame, factor returns [Mkt-RF, SMB, HML]
                       (Mkt-RF is an excess return; SMB/HML are zero-cost spreads)

Key functions:
    time_series_regressions  : per-portfolio time-series regression (with HAC t-stats)
    grs_test                 : Gibbons-Ross-Shanken (1989) joint test
    summarize                : bundles the two into a comparison table + one GRS line
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def _nw_lag(T: int) -> int:
    """Newey-West (1994) automatic lag rule: floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def time_series_regressions(excess_returns: pd.DataFrame,
                            factors: pd.DataFrame,
                            hac: bool = True,
                            hac_lags: int | None = None) -> dict:
    """
    For each portfolio i, fit
        R_i - RF = alpha_i + b*(Mkt-RF) + s*SMB + h*HML + eps

    If hac=True, the t-stat of alpha_i uses Newey-West (HAC) standard errors.
    If hac_lags=None, the automatic _nw_lag(T) rule is used.

    Returns dict:
        alpha   (N,)      intercepts
        beta    (N x K)   factor loadings
        resid   (T x N)   residuals (GRS input)
        alpha_t (N,)      t-stat of alpha (HAC or OLS)
        r2      (N,)       R-squared
        T, N, K           sample dimensions
        names             list of portfolio names
    """
    Y = excess_returns.values
    F = factors.values
    T, N = Y.shape
    K = F.shape[1]

    X = sm.add_constant(F)  # T x (K+1), first column is the intercept

    alpha = np.empty(N)
    beta = np.empty((N, K))
    resid = np.empty((T, N))
    alpha_t = np.empty(N)
    r2 = np.empty(N)

    L = hac_lags if hac_lags is not None else _nw_lag(T)

    for i in range(N):
        if hac:
            res = sm.OLS(Y[:, i], X).fit(cov_type="HAC",
                                         cov_kwds={"maxlags": L})
        else:
            res = sm.OLS(Y[:, i], X).fit()
        alpha[i] = res.params[0]
        beta[i] = res.params[1:]
        resid[:, i] = res.resid
        alpha_t[i] = res.tvalues[0]
        r2[i] = res.rsquared

    return {
        "alpha": alpha, "beta": beta, "resid": resid,
        "alpha_t": alpha_t, "r2": r2,
        "T": T, "N": N, "K": K,
        "names": list(excess_returns.columns),
        "hac_lags": L,
    }


def grs_test(alpha: np.ndarray,
             resid: np.ndarray,
             factors: np.ndarray) -> dict:
    """
    Gibbons-Ross-Shanken (1989) joint test.
    Null hypothesis: the N intercepts (alpha) are jointly zero.

    Statistic (exactly F in finite samples under normality):
        J = ((T - N - K) / N) * (a' Sigma^{-1} a) / (1 + mu' Omega^{-1} mu)
        J ~ F(N, T - N - K)

    where
        Sigma = (1/T) * resid' resid     (residual covariance, ML estimator, divisor T)
        Omega = (1/T) * (f-mu)'(f-mu)     (factor covariance, ML estimator, divisor T)
        mu    = factor sample means
    The divisor must be T so the degrees of freedom are exactly (N, T-N-K).

    Returns dict: F, p_value, dof1(=N), dof2(=T-N-K),
                  sharpe2(= mu' Omega^{-1} mu, squared max Sharpe of the factors)
    """
    R = np.asarray(resid)
    f = np.asarray(factors)
    a = np.asarray(alpha)

    T, N = R.shape
    K = f.shape[1]

    mu = f.mean(axis=0)                      # (K,)
    fd = f - mu
    Omega = (fd.T @ fd) / T                  # (K x K)
    Sigma = (R.T @ R) / T                    # (N x N)

    Sigma_inv = np.linalg.inv(Sigma)
    Omega_inv = np.linalg.inv(Omega)

    sharpe2 = float(mu @ Omega_inv @ mu)     # squared max Sharpe attainable from the factors
    quad = float(a @ Sigma_inv @ a)

    dof2 = T - N - K
    F = ((dof2) / N) * quad / (1.0 + sharpe2)
    p = float(stats.f.sf(F, N, dof2))

    return {"F": float(F), "p_value": p, "dof1": N, "dof2": dof2,
            "sharpe2": sharpe2}


def summarize(excess_returns: pd.DataFrame,
              factors: pd.DataFrame,
              hac: bool = True,
              hac_lags: int | None = None):
    """
    Shared entry point for the US validation and the Korea replication.
    Returns a per-portfolio (alpha, alpha_t, r2) table together with one GRS line.

    Returns: (table_df, grs_dict, reg_dict)
    """
    reg = time_series_regressions(excess_returns, factors,
                                  hac=hac, hac_lags=hac_lags)
    grs = grs_test(reg["alpha"], reg["resid"], factors.values)

    table = pd.DataFrame({
        "portfolio": reg["names"],
        "alpha": reg["alpha"],
        "alpha_t": reg["alpha_t"],
        "r2": reg["r2"],
    })

    grs = {
        **grs,
        "mean_abs_alpha": float(np.mean(np.abs(reg["alpha"]))),
        "hac_lags": reg["hac_lags"],
    }
    return table, grs, reg
