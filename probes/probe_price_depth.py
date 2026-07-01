# probe_price_depth.py
# ─────────────────────────────────────────────────────────────────────────
# Return-source depth — can value-weighted portfolio monthly returns be computed back to 2006?
#
# Background: in pykrx_probe, get_market_ohlcv_by_date (single-ticker time series,
#       Naver source) is capped at ~2014 / 3000 rows. But factor returns need
#       month-end cross-sectional closing prices, which come from
#       get_market_ohlcv (cross-section, KRX login) — the same source as the
#       market-cap cross-section that goes back to 1995, so it is likely deep.
#       If this is capped at 2014, returns are stuck at 2014 -> the 2008
#       sub-period falls outside the sample and cannot be tested. Hence this check.
#
# Run: python probes/probe_price_depth.py
# ─────────────────────────────────────────────────────────────────────────

# %% [0] login (.env first) + imports
import os, warnings
from dotenv import load_dotenv
load_dotenv()
from pykrx import stock
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %% [1] cross-sectional close validity + walk-back
def valid_close(date_str, market):
    """(total stocks, stocks with close>0). Failure -> (None, None); empty -> (0, 0)."""
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

# %% [2] month-end cross-sectional close depth — June/Dec 2003-2015
def scan(label, dates, market):
    rows = [resolve_close(t, market) for t in dates]
    df = pd.DataFrame(rows)[["target", "used", "walked", "n_close", "ok"]]
    print(f"\n=== {label} ({market}) ===")
    print(df.to_string(index=False))
    first_ok = df[df["ok"]]["target"].min() if df["ok"].any() else None
    print(f"  first valid month-end close: {first_ok}")
    return df, first_ok

jun = [f"{y}0630" for y in range(2003, 2016)]
dec = [f"{y}1230" for y in range(2002, 2016)]

kj, kj0 = scan("June close", jun, "KOSPI")
kd, kd0 = scan("Dec close", dec, "KOSPI")
qj, qj0 = scan("June close", jun, "KOSDAQ")
qd, qd0 = scan("Dec close", dec, "KOSDAQ")

# %% [3] cap/shares fallback check — if the cross-sectional OHLCV is shallow, can price be proxied from cap/shares?
cap08 = stock.get_market_cap("20080630", market="KOSPI")
has_shares = ("상장주식수" in cap08.columns) and (cap08["상장주식수"] > 0).any()
print(f"\n[fallback] 2008-06 cap cross-section: {len(cap08)} stocks · shares valid={has_shares}")
print("  → Even if the cross-sectional OHLCV is shallow, if shares outstanding are present, price ≈ cap/shares can proxy returns (dividends handled separately).")

# %% [4] verdict
print("\n" + "=" * 60)
print("verdict — the return start date ultimately fixes the sample window")
print("=" * 60)
print(f"KOSPI first valid close: June={kj0} · Dec={kd0}")
print(f"KOSDAQ first valid close: June={qj0} · Dec={qd0}")
print()
print("→ If KOSPI+KOSDAQ closes start before 2006: keep the primary sample from 2006-07, include 2008, and proceed with extraction.")
print("→ If closes are capped at 2014: returns stuck at 2014 → 2008 outside the sample → redesign the 2008 test + shrink the window. (See fallback [3].)")
