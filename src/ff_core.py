"""
ff_core.py — Fama-French 검증 엔진 (데이터 소스 무관)

이 모듈은 P1 v0.1(미국 검증)과 P2(한국 복제)에서 공통으로 쓰인다.
입력 규약만 맞추면 데이터 출처와 상관없이 동작한다:
    - excess_returns : (T x N) DataFrame, 포트폴리오 i의 초과수익 (R_i - RF)
    - factors        : (T x K) DataFrame, 팩터 수익률 [Mkt-RF, SMB, HML]
                       (Mkt-RF는 초과수익, SMB/HML은 무비용 스프레드)

핵심 함수:
    time_series_regressions  : 포트폴리오별 시계열 회귀 (HAC t값 포함)
    grs_test                 : Gibbons-Ross-Shanken (1989) 결합검정
    summarize                : 위 둘을 묶어 대조용 표 + GRS 한 줄로 반환
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def _nw_lag(T: int) -> int:
    """Newey-West(1994) 자동 시차 규칙: floor(4 * (T/100)^(2/9))."""
    return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def time_series_regressions(excess_returns: pd.DataFrame,
                            factors: pd.DataFrame,
                            hac: bool = True,
                            hac_lags: int | None = None) -> dict:
    """
    각 포트폴리오 i에 대해
        R_i - RF = alpha_i + b*(Mkt-RF) + s*SMB + h*HML + eps
    를 적합한다.

    hac=True 이면 alpha_i의 t값을 Newey-West(HAC) 표준오차로 계산한다.
    hac_lags=None 이면 _nw_lag(T) 자동 규칙을 쓴다.

    반환 dict:
        alpha   (N,)      절편
        beta    (N x K)   팩터 적재
        resid   (T x N)   잔차 (GRS 입력)
        alpha_t (N,)      alpha의 t값 (HAC 또는 OLS)
        r2      (N,)       결정계수
        T, N, K           표본 차원
        names             포트폴리오 이름 리스트
    """
    Y = excess_returns.values
    F = factors.values
    T, N = Y.shape
    K = F.shape[1]

    X = sm.add_constant(F)  # T x (K+1), 1열은 절편

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
    Gibbons-Ross-Shanken (1989) 결합검정.
    귀무가설: N개 절편(alpha)이 동시에 0.

    통계량 (정규성 가정하에 유한표본에서 정확히 F):
        J = ((T - N - K) / N) * (a' Sigma^{-1} a) / (1 + mu' Omega^{-1} mu)
        J ~ F(N, T - N - K)

    여기서
        Sigma = (1/T) * resid' resid     (잔차 공분산, ML 추정량, 나눗수 T)
        Omega = (1/T) * (f-mu)'(f-mu)     (팩터 공분산, ML 추정량, 나눗수 T)
        mu    = 팩터 표본평균
    나눗수를 T로 맞춰야 자유도가 정확히 (N, T-N-K)가 된다.

    반환 dict: F, p_value, dof1(=N), dof2(=T-N-K),
              sharpe2(= mu' Omega^{-1} mu, 팩터 최대 샤프비 제곱)
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

    sharpe2 = float(mu @ Omega_inv @ mu)     # 팩터로 달성 가능한 최대 샤프비^2
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
    v0.1/P2 공용 진입점.
    포트폴리오별 (alpha, alpha_t, r2) 표와 GRS 한 줄을 함께 반환한다.

    반환: (table_df, grs_dict, reg_dict)
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
