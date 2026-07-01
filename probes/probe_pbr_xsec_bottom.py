# probe_pbr_xsec_bottom.py
# ─────────────────────────────────────────────────────────────────────────
# Robust cross-sectional fundamental fetcher + rebalance re-scan
#
# Why: an earlier validation check found that get_market_fundamental returns a
#     blank ("") on certain dates, and pykrx substitutes that with 0 (wrap.py:249).
#     If the June-end / December-end dates are hard-coded, the B/M for that
#     rebalance year is silently emptied with no error raised. Judging the trading
#     day from market cap does not catch it, because cap and fundamental can
#     disagree -> instead, walk back based on the fundamental's own emptiness.
#
# This fetcher is not throwaway: it is promoted directly to the date resolver
# used by the data-extraction stage.
#
# Run: python probes/probe_pbr_xsec_bottom.py
# ─────────────────────────────────────────────────────────────────────────

# %% [0] login (.env first) + imports + silence warnings
import os, warnings
from dotenv import load_dotenv
load_dotenv()
from pykrx import stock
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)   # harmless replace("",0) downcasting warning

# %% [1] robust fetcher — walk backward based on the fundamental actually being populated
def resolve_and_fetch(target, market, max_walk=15, min_count=50):
    """Walk backward from target (YYYYMMDD) and return the fundamental of the
       first date with (number of stocks with PBR>0) >= min_count. Blanks (->0),
       market holidays, and empty frames are skipped automatically.
       Returns: dict(target, used, walked, n_valid, ok, frame)."""
    d = pd.Timestamp(target)
    for k in range(max_walk + 1):
        ds = d.strftime("%Y%m%d")
        try:
            f = stock.get_market_fundamental(ds, market=market)
        except Exception:
            f = None
        n = 0 if (f is None or len(f) == 0) else int((f["PBR"] > 0).sum())
        if n >= min_count:
            return {"target": target, "used": ds, "walked": k, "n_valid": n, "ok": True, "frame": f}
        d -= pd.Timedelta(days=1)
    return {"target": target, "used": None, "walked": max_walk, "n_valid": 0, "ok": False, "frame": None}

# %% [2] re-scan — June rebalance (size, June) 2003-2012 · Dec rebalance (B/M, December ME) 2002-2011
#   If walked is small (<= a week), the sample floor is confirmed at 2003 and the
#   extraction stage just needs the walk-back attached.
#   If a given year is ok=False or walked is large, that year's cross-section is
#   genuinely sparse -> a by-ticker assembly is needed.
def scan(label, dates, market):
    rows = [resolve_and_fetch(t, market) for t in dates]
    df = pd.DataFrame(rows)[["target", "used", "walked", "n_valid", "ok"]]
    print(f"\n=== {label} ({market}) ===")
    print(df.to_string(index=False))
    bad = df[~df["ok"]]
    farr = df[df["walked"] > 7]
    if len(bad):
        print(f"  ⚠ resolve failed (no valid date within 15 days): {bad['target'].tolist()} → by-ticker assembly candidate for that year")
    if len(farr):
        print(f"  ⚠ walk-back >= 8 days: {list(zip(farr['target'], farr['walked']))} → check the date")
    if len(bad) == 0 and len(farr) == 0:
        print("  ✅ all rebalances resolved within a 7-day walk-back → this market's floor / endpoint OK")
    return df

jun = [f"{y}0630" for y in range(2003, 2013)]
dec = [f"{y}1230" for y in range(2002, 2012)]

kospi_jun  = scan("June rebalance (size)",   jun, "KOSPI")
kospi_dec  = scan("Dec rebalance (B/M ME)",  dec, "KOSPI")
kosdaq_jun = scan("June rebalance (size)",   jun, "KOSDAQ")
kosdaq_dec = scan("Dec rebalance (B/M ME)",  dec, "KOSDAQ")

# %% [3] summary — determine the actual sample floor
print("\n" + "=" * 60)
print("verdict")
print("=" * 60)
for name, dfj, dfd in [("KOSPI", kospi_jun, kospi_dec), ("KOSDAQ", kosdaq_jun, kosdaq_dec)]:
    okj = dfj[dfj["ok"]]["target"].min() if dfj["ok"].any() else None
    okd = dfd[dfd["ok"]]["target"].min() if dfd["ok"].any() else None
    print(f"{name}: first valid June={okj} · first valid Dec={okd}")
print("\n→ If every rebalance is ok=True with a small walk-back, keep the sample floor at 2003 and the B/M source / 2008 sub-period as-is.")
print("→ If a given year is ok=False, only that year becomes a by-ticker assembly target (footnote + a branch in the extraction stage).")
print("→ Either way, promote resolve_and_fetch to the extraction date resolver (log used · walked at every rebalance).")
