#!/usr/bin/env python3
"""regime_tests.py — H3용 구간 평균차 검정 (표 3의 구간 정의 그대로).

주 패널(VW) 팩터 월별 시계열에 구간 더미 회귀 + HAC(Bartlett) 표준오차를
적용해 세 가지를 검정한다. HAC 시차는 엔진(ff_core._nw_lag)의 자동 규칙을
그대로 쓴다.

  (a) 핵심 대비   : GFC 대 GFC 이후 평균차 (표본 2007-07~2026-06, 더미 t)
  (b) 3구간 동일성 : 위기 이전 = GFC = 이후 (Wald, 자유도 2)
  (c) 5구간 동일성 : (b) + 2020·2022 연도 더미 (Wald, 자유도 4; 참고용)

구간 정의(표 3과 동일):
  위기 이전 2006-07~2007-06 · GFC 2007-07~2010-06 · 이후 2010-07~2026-06
  2020-01~2020-12 · 2022-01~2022-12

위치·실행: src/regime_tests.py 로 두고, 어느 폴더에서든
    python src/regime_tests.py
산출: results/regime_tests.csv + 화면 요약.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "src" else HERE
RESULTS = ROOT / "results"
sys.path.insert(0, str(HERE))

try:
    from ff_core import _nw_lag as nw_lag  # noqa: E402
except Exception:
    def nw_lag(T: int) -> int:
        """Newey-West (1994) automatic lag rule: floor(4 * (T/100)^(2/9))."""
        return int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))

REGIMES = {
    "pre":  (200607, 200706),
    "gfc":  (200707, 201006),
    "post": (201007, 202606),
    "y20":  (202001, 202012),
    "y22":  (202201, 202212),
}


def hac_ols(y: np.ndarray, X: np.ndarray):
    """OLS 계수 + Bartlett HAC 공분산 (엔진과 동일 커널·시차 규칙)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    T, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    L = nw_lag(T)
    xe = X * e[:, None]
    S = xe.T @ xe / T
    for l in range(1, L + 1):
        G = xe[l:].T @ xe[:-l] / T
        S += (1.0 - l / (L + 1.0)) * (G + G.T)
    V = XtX_inv @ (T * S) @ XtX_inv
    return b, V, L


def wald(b, V, idx):
    """H0: b[idx] = 0 (동시). Wald 통계량과 자유도."""
    R = np.zeros((len(idx), len(b)))
    for r, i in enumerate(idx):
        R[r, i] = 1.0
    Rb = R @ b
    W = float(Rb @ np.linalg.inv(R @ V @ R.T) @ Rb)
    return W, len(idx)


def p_norm2(t):
    """양측 정규 p."""
    return math.erfc(abs(t) / math.sqrt(2.0))


def p_chi2(x, df):
    """카이제곱 우측꼬리 p (본 스크립트가 쓰는 자유도만)."""
    if df == 1:
        return math.erfc(math.sqrt(x / 2.0))
    if df == 2:
        return math.exp(-x / 2.0)
    if df == 4:
        return math.exp(-x / 2.0) * (1.0 + x / 2.0)
    raise ValueError(f"df={df} 미지원")


def main():
    kr = pd.read_csv(RESULTS / "factors_prim_vw.csv")
    kr["ym"] = pd.to_datetime(kr["date"]).dt.strftime("%Y%m").astype(int)
    ym = kr["ym"].values
    masks = {k: (ym >= a) & (ym <= b) for k, (a, b) in REGIMES.items()}

    rows = []
    for f in ["HML", "SMB", "MKT-RF"]:
        x = kr[f].values * 100.0  # 소수 -> %/월
        print(f"\n===== {f} =====")
        means = {k: x[m].mean() for k, m in masks.items()}
        print("구간 평균(%/월): " + " | ".join(f"{k} {v:+.2f}" for k, v in means.items()))

        # (a) GFC 대 이후
        m = masks["gfc"] | masks["post"]
        X = np.column_stack([np.ones(m.sum()), masks["gfc"][m].astype(float)])
        b, V, L = hac_ols(x[m], X)
        t = b[1] / math.sqrt(V[1, 1])
        p = p_norm2(t)
        print(f"(a) GFC-이후 평균차 {b[1]:+.2f}%p   HAC t={t:+.2f}   p={p:.3f}   (T={m.sum()}, lag={L})")
        rows.append({"factor": f, "test": "a_gfc_vs_post", "estimate_pp": round(b[1], 3),
                     "stat": round(t, 2), "df": 1, "p": round(p, 4), "T": int(m.sum()), "lag": L})

        # (b) 3구간 동일성
        X = np.column_stack([np.ones(len(x)),
                             masks["pre"].astype(float), masks["gfc"].astype(float)])
        b, V, L = hac_ols(x, X)
        W, df = wald(b, V, [1, 2])
        p = p_chi2(W, df)
        print(f"(b) 3구간 동일성        Wald χ²({df})={W:.2f}   p={p:.3f}")
        rows.append({"factor": f, "test": "b_joint3", "estimate_pp": None,
                     "stat": round(W, 2), "df": df, "p": round(p, 4), "T": len(x), "lag": L})

        # (c) 5구간 동일성 (2020·2022 참고 포함)
        X = np.column_stack([np.ones(len(x)),
                             masks["pre"].astype(float), masks["gfc"].astype(float),
                             masks["y20"].astype(float), masks["y22"].astype(float)])
        b, V, L = hac_ols(x, X)
        W, df = wald(b, V, [1, 2, 3, 4])
        p = p_chi2(W, df)
        ts = [b[i] / math.sqrt(V[i, i]) for i in range(1, 5)]
        print(f"(c) 5구간 동일성(참고)   Wald χ²({df})={W:.2f}   p={p:.3f}   "
              f"[개별 t: pre {ts[0]:+.2f} gfc {ts[1]:+.2f} 2020 {ts[2]:+.2f} 2022 {ts[3]:+.2f}]")
        rows.append({"factor": f, "test": "c_joint5_ref", "estimate_pp": None,
                     "stat": round(W, 2), "df": df, "p": round(p, 4), "T": len(x), "lag": L})

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "regime_tests.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[regime_tests] 저장: {out}")


if __name__ == "__main__":
    main()
