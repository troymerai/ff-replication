# probe_price_depth.py
# ─────────────────────────────────────────────────────────────────────────
# 수익률 소스 깊이 — VW 포트폴리오 월수익률이 2006까지 계산되는가
#
# 배경: pykrx_probe에서 get_market_ohlcv_by_date(단일종목 시계열, Naver)가 2014~/3000행 캡.
#       하지만 팩터 수익률은 월말 횡단면 종가가 필요하고, 그건 get_market_ohlcv(횡단면, KRX 로그인)
#       경로다 — 시총 횡단면이 1995까지인 것과 같은 소스라 깊을 가능성이 높다.
#       이게 2014에서 막히면 수익률이 2014 고정 → 2008 표본 밖 → H3 사망. 그래서 확인.
#
# 실행: & "D:\ff-replication\venv\Scripts\python.exe" "D:\ff-replication\tests\probe_price_depth.py"
# ─────────────────────────────────────────────────────────────────────────

# %% [0] 로그인(.env 먼저) + import
import os, warnings
from dotenv import load_dotenv
load_dotenv()
from pykrx import stock
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %% [1] 횡단면 종가 유효성 + walk-back
def valid_close(date_str, market):
    """(총 종목수, 종가>0 종목수). 실패 (None,None) / 빈 (0,0)."""
    try:
        o = stock.get_market_ohlcv(date_str, market=market)
    except Exception:
        return None, None
    if o is None or len(o) == 0:
        return 0, 0
    col = "종가" if "종가" in o.columns else o.columns[3]
    return len(o), int((o[col] > 0).sum())

def resolve_close(target, market, max_walk=15, min_count=50):
    d = pd.Timestamp(target)
    for k in range(max_walk + 1):
        ds = d.strftime("%Y%m%d")
        n_tot, n = valid_close(ds, market)
        if n is not None and n >= min_count:
            return {"target": target, "used": ds, "walked": k, "n_close": n, "ok": True}
        d = d - pd.Timedelta(1, "D")
    return {"target": target, "used": None, "walked": max_walk, "n_close": 0, "ok": False}

# %% [2] 월말 횡단면 종가 깊이 — June/Dec 2003~2015
def scan(label, dates, market):
    rows = [resolve_close(t, market) for t in dates]
    df = pd.DataFrame(rows)[["target", "used", "walked", "n_close", "ok"]]
    print(f"\n=== {label} ({market}) ===")
    print(df.to_string(index=False))
    first_ok = df[df["ok"]]["target"].min() if df["ok"].any() else None
    print(f"  유효 첫 월말 종가: {first_ok}")
    return df, first_ok

jun = [f"{y}0630" for y in range(2003, 2016)]
dec = [f"{y}1230" for y in range(2002, 2016)]

kj, kj0 = scan("June 종가", jun, "KOSPI")
kd, kd0 = scan("Dec 종가", dec, "KOSPI")
qj, qj0 = scan("June 종가", jun, "KOSDAQ")
qd, qd0 = scan("Dec 종가", dec, "KOSDAQ")

# %% [3] cap/shares 폴백 확인 — 횡단면 OHLCV가 얕아도 시총/주식수로 가격 프록시 가능한지
cap08 = stock.get_market_cap("20080630", market="KOSPI")
has_shares = ("상장주식수" in cap08.columns) and (cap08["상장주식수"] > 0).any()
print(f"\n[폴백] 2008-06 시총 횡단면: {len(cap08)}종목 · 상장주식수 유효={has_shares}")
print("  → 횡단면 OHLCV가 얕더라도 상장주식수가 있으면 price ≈ 시총/주식수 로 수익률 프록시 가능(배당은 별도).")

# %% [4] 판정
print("\n" + "=" * 60)
print("판정 — 수익률 시작 시점이 표본 창을 최종 확정")
print("=" * 60)
print(f"KOSPI 종가 유효 첫 시점: June={kj0} · Dec={kd0}")
print(f"KOSDAQ 종가 유효 첫 시점: June={qj0} · Dec={qd0}")
print()
print("→ KOSPI+KOSDAQ 종가가 2006 이전부터면: 주 표본 2006-07~ 그대로 확정, 2008(H3) 포함, 5a 진행.")
print("→ 종가가 2014에서 막히면: 수익률 2014 고정 → 2008 표본 밖 → H3 재설계 + 창 축소. (폴백 [3] 검토)")
